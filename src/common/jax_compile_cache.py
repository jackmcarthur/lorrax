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

Default location is ``~/.cache/isdf_jax_compilation``.

At MoS2 3x3 scale this cache is ~267 entries / 1.9 MB on disk and
saves ~3-4 s of XLA compile on warm re-runs (measured 2026-04-19).

The env-var naming is legacy — historically this was for ISDF
kernels only, now it caches the whole run.  Left as-is for backward
compat with existing user shell aliases and run scripts.
"""
from __future__ import annotations

import os
from pathlib import Path

_COMPILATION_CACHE_READY = False


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

    Idempotent: safe to call multiple times.  Calling from two
    different drivers in the same process is fine.
    """
    global _COMPILATION_CACHE_READY
    if _COMPILATION_CACHE_READY:
        return
    cache_dir = os.environ.get("ISDF_JAX_CACHE_DIR")
    if cache_dir is None:
        base_cache = os.environ.get(
            "XDG_CACHE_HOME", os.path.expanduser("~/.cache"))
        cache_dir = os.path.join(base_cache, "isdf_jax_compilation")
    if not cache_dir:  # explicit opt-out via ISDF_JAX_CACHE_DIR=""
        _COMPILATION_CACHE_READY = True
        return
    # Namespace cache by num_processes.  JAX's persistent cache
    # hashes device assignment into the key; a run with different
    # process count has different device assignment and the entries
    # wouldn't be re-usable anyway.  Within a given n_proc we let all
    # ranks share a dir — in SPMD patterns (gw_jax with shard_map)
    # every rank has an identical cache key so sharing lets each
    # rank read any rank's saved entries; in k-parallel DP patterns
    # (psp.run_nscf, rank N owns local GPU N) JAX writes entries
    # only on rank 0 anyway, so sharing is equivalent to per-rank.
    # The DP case may emit "Device assignment does not have any
    # local devices" warnings when ranks 1-3 try to load rank 0's
    # entries — those warnings are non-fatal (the compile proceeds),
    # cost ~ms each, and only happen on one warm run per cache.
    try:
        import jax as _jax
        n_proc = _jax.process_count()
    except Exception:
        n_proc = 1
    cache_path = Path(cache_dir).expanduser() / f"np{n_proc}"
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


# Backward compat alias.
_ensure_compilation_cache = ensure_jax_compile_cache
