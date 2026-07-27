"""JAX persistent compile cache — one-shot activator, shared across drivers.

Every LORRAX driver that does a meaningful amount of ``jax.jit``
compile (gw.gw_jax, psp.run_nscf, centroid.kmeans_isdf, bse.bse_jax,
etc.) should call :func:`ensure_jax_compile_cache` near the top of
its ``main()`` — right after ``jax.distributed.initialize`` and the
mesh construction, BEFORE any jit.  Calling it late means the
expensive compiles that happen early in the pipeline miss the cache.

Knobs:

  ``ISDF_JAX_CACHE_DIR=/some/path``   — override cache location.
  ``ISDF_JAX_CACHE_DIR=""``            — opt out entirely.
  ``LORRAX_JAX_CACHE_MULTIPROCESS=1``  — re-arm the cache at P > 1 (see
                                         the deadlock note below).  OFF.

Default location is ``~/.cache/isdf_jax_compilation``.

At MoS2 3x3 scale this cache is ~267 entries / 1.9 MB on disk and
saves ~3-4 s of XLA compile on warm re-runs (measured 2026-04-19).

The env-var naming is legacy — historically this was for ISDF
kernels only, now it caches the whole run.  Left as-is for backward
compat with existing user shell aliases and run scripts.

WHY THE CACHE IS OFF AT ``jax.process_count() > 1`` (scorecard AG)
=================================================================
This is the ``load_centroid_wfns`` hang that blocked every multi-process
GPU run on Frontera's ``rtx`` queue.  The mechanism, root-caused from
a live C-level stack (workstream AG, job 7876375):

* JAX writes persistent-cache entries **from process 0 only** — not a
  LORRAX choice, it is unconditional in
  ``jax/_src/compiler.py::_cache_write``::

      # Only write cache entries from the first process. Otherwise we
      # create problems with contention for writes on some filesystems
      if distributed.global_state.process_id != 0:
        return

* This module used to partition the cache **per rank**
  (``{base}/np{P}/rank{i}/``) so each rank only ever saw its own
  entries.  Combined with the above that is a permanent asymmetry:
  ``rank0/`` accumulates entries run after run (measured: 882) while
  ``rank1..P-1/`` stay **empty forever**.

* On the next run process 0 therefore HITS the cache and skips
  compilation, while every peer MISSES and compiles.  On XLA:GPU
  compilation is not a local act: ``xla::gpu::AutotunerPass`` shards the
  autotuning across processes and exchanges the results through the JAX
  coordination service (``xla::Autotuner::Autotune(HloModule*, ...,
  MultiProcessKeyValueStore&)`` → ``BlockingKeyValueGet``).  A process
  that skipped compilation never publishes its share, so the peers block
  in ``CoordinationServiceAgent::GetKeyValue`` **forever**.

The observable is a silent hang with one rank spinning at ~70-100 % CPU
(process 0, already downstream and waiting in a collective), the rest
parked at ~2 % CPU, and every GPU at 0 % utilisation.  It cost three
GPU jobs and a workstream to find, which is exactly why the refusal
below is LOUD.

XLA:CPU has no such autotuner, but the campaign's CPU launchers all
export ``ISDF_JAX_CACHE_DIR=""`` by hand anyway; this guard makes that
out-of-band convention an in-tree property so a launcher that forgets
it (``config/frontera/ffi_env.sh`` did) cannot resurrect the hang.

Making the cache genuinely safe at P > 1 needs process-0-writes to be
readable by every rank *and* the resulting hit/miss pattern to be
identical on every rank — a single shared dir does not achieve the
second (JAX's cache key hashes the per-rank device assignment, so the
peers still miss).  Until that is solved, P > 1 runs pay the compile.
"""
from __future__ import annotations

import os
import warnings
from pathlib import Path

_COMPILATION_CACHE_READY = False


def _truthy(name: str, default: str = "0") -> bool:
    return os.environ.get(name, default).strip().lower() not in (
        "", "0", "false", "no", "off")


def ensure_jax_compile_cache() -> None:
    """Enable JAX persistent compile cache once per process.

    Uses the modern ``jax.config.update('jax_compilation_cache_dir', ...)``
    API.  The legacy ``jax.experimental.compilation_cache.set_cache_dir``
    is soft-deprecated and silently no-ops on recent jaxlib, so we
    don't rely on it (falls back only if the modern config knob isn't
    present).  Also explicitly sets
    ``jax_persistent_cache_min_compile_time_secs=0.0`` so every
    compile JAX deems worth caching lands on disk — JAX internally
    skips truly trivial compiles (const, simple reshape, etc.), so
    "cache everything" is already reasonably targeted.

    **Single-process runs only, by default.**  At
    ``jax.process_count() > 1`` this function REFUSES to arm the cache
    and says so on process 0 — see the module docstring for the
    autotuner deadlock that refusal prevents (scorecard AG).
    ``LORRAX_JAX_CACHE_MULTIPROCESS=1`` re-arms it for whoever fixes
    the underlying asymmetry.

    Partitioning (single-process path, and the P>1 path if re-armed):
    cache entries are nested under ``{base}/np{N_proc}/rank{N}/`` so
    each rank reads/writes its own dir.  JAX's cache key hashes in the
    per-rank device assignment, which means rank 1 would otherwise try
    to load rank 0's entries, find a device-index mismatch, and emit a
    ``"Device assignment does not have any local devices"`` warning
    per primitive per JIT on every warm run.
    NOTE the sting in the tail, and why per-rank dirs are not a fix at
    P > 1: JAX writes cache entries from **process 0 only**, so the
    peers' dirs never fill and the "per-rank caches converge within one
    warm run" claim this docstring used to make is **false** — measured
    ``np4/rank0`` = 882 entries, ``np4/rank{1,2,3}`` = 0, three days
    after those dirs were created.

    Also silences the legacy warning via a targeted
    ``warnings.filterwarnings`` as defense-in-depth for cases where
    a pre-existing shared cache (e.g. from older versions of this
    helper) is still on disk.

    Idempotent: safe to call multiple times.  Calling from two
    different drivers in the same process is fine.
    """
    global _COMPILATION_CACHE_READY
    if _COMPILATION_CACHE_READY:
        return

    # Silence "Error reading persistent compilation cache entry ...
    # Device assignment does not have any local devices" — emitted by
    # jax/_src/compiler.py when it finds a cache entry whose device
    # binding doesn't match this rank.  Non-fatal; JAX recompiles.
    # Filter is global but message-specific so other warnings are
    # untouched.
    warnings.filterwarnings(
        "ignore",
        message=r"Error reading persistent compilation cache entry .*",
        category=UserWarning,
    )

    cache_dir = os.environ.get("ISDF_JAX_CACHE_DIR")
    if cache_dir is None:
        base_cache = os.environ.get(
            "XDG_CACHE_HOME", os.path.expanduser("~/.cache"))
        cache_dir = os.path.join(base_cache, "isdf_jax_compilation")
    if not cache_dir:  # explicit opt-out via ISDF_JAX_CACHE_DIR=""
        _COMPILATION_CACHE_READY = True
        return
    try:
        import jax as _jax
        n_proc = _jax.process_count()
        proc_idx = _jax.process_index()
    except Exception:
        n_proc = 1
        proc_idx = 0

    # ---- the P > 1 refusal (see the module docstring for the mechanism) ----
    if n_proc > 1 and not _truthy("LORRAX_JAX_CACHE_MULTIPROCESS"):
        if proc_idx == 0:
            print(
                f"  [compile-cache] REFUSED at {n_proc} processes: the JAX "
                f"persistent compile cache is written by process 0 ONLY "
                f"(jax/_src/compiler.py::_cache_write), so on a multi-process "
                f"run the ranks' hit/miss patterns diverge — process 0 skips "
                f"compilation while its peers block forever in "
                f"xla::gpu::AutotunerPass' cross-process key-value exchange "
                f"(CoordinationServiceAgent::GetKeyValue).  That is the "
                f"silent `load_centroid_wfns` hang of scorecard AG.\n"
                f"  [compile-cache] this run compiles from scratch on every "
                f"rank, which is CORRECT and ~1 min slower.  Set "
                f"ISDF_JAX_CACHE_DIR=\"\" to silence this line, or "
                f"LORRAX_JAX_CACHE_MULTIPROCESS=1 to re-arm the cache and "
                f"accept the hang (would have used {cache_dir}).",
                flush=True)
        _COMPILATION_CACHE_READY = True
        return

    cache_path = (Path(cache_dir).expanduser()
                  / f"np{n_proc}" / f"rank{proc_idx}")
    try:
        cache_path.mkdir(parents=True, exist_ok=True)
    except Exception:
        _COMPILATION_CACHE_READY = True
        return
    try:
        import jax as _jax
        _jax.config.update("jax_compilation_cache_dir", str(cache_path))
        _jax.config.update(
            "jax_persistent_cache_min_compile_time_secs", 0.0)
    except Exception:
        try:
            from jax.experimental import (
                compilation_cache as _legacy_cache)
            _legacy_cache.set_cache_dir(str(cache_path))
        except Exception:
            pass
    _COMPILATION_CACHE_READY = True
