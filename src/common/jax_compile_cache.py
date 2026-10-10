"""JAX's persistent compile cache, armed once per process by the runtime.

JAX 0.9.1 does the caching: ``jax_compilation_cache_dir`` names the store,
the key hashes the module, the compile options, XLA flags, jaxlib and the
backend version, process 0 is the only writer, and on GPU the key is the
same on every rank (the device assignment is stripped). What this file adds
is only what that cannot see or say:

* the location: ``$SCRATCH/.cache/lorrax/jax_compile/<namespace>/np{P}``
  (:func:`default_cache_dir`), where the namespace names jax, jaxlib, the
  native FFI bundle and this file's key schema, never the LORRAX source (a
  commit reaches a program only through its HLO, which the key hashes);
  rank 0 touches the entries it uses and retires the unused after a week
  (:func:`_prune_namespaces`). ``ISDF_JAX_CACHE_DIR`` overrides, ``""``
  opts out (``docs/reference/env_vars.md``);
* a threshold of 0 s (JAX's 1 s default kept 2 of 666 MoS2 executables), and
  JAX's own per-fusion XLA caches off at P > 1 (they are rank-asymmetric,
  UPDATE on process 0 and READ on its peers);
* the cross-rank compile agreement (:func:`install_compile_agreement`): a
  rank that lowers a different program than its peers is refused by name
  before the backend compile instead of hanging in its next collective
  (INVARIANTS 21), with the compile counter the receipt reads;
* GATE xla_rematerialization (:func:`_install_device_fit_gate`): a module
  larger than the device is refused before its first execution;
* :func:`compile_ahead`: a program whose avals a stage owner knows before its
  first call is lowered on the calling thread (program order, so every rank
  lowers the same module at the same request number) and compiled on a
  helper thread, while the main thread goes on tracing, lowering and running.
  XLA's backend compile releases the GIL and scales with threads (sandbox
  claims 4178, 4180); the agreement slot is taken on the calling thread, so
  INVARIANTS 25's race (helper compiles reordering the requests across
  ranks) cannot arise; a live call for a module in flight waits for that
  compile instead of starting a second one.

What this file no longer does, and why. Until 2026-10-05 it froze an
all-rank agreed entry set at startup and vetoed every other lookup, made
the key process-invariant on CPU, canonicalized ``jit__multi_slice``, wrote
entries atomically and prefetched them: all against XLA:GPU's collective
autotuner, which hung when one rank hit the cache and skipped the exchange
(7648fd417, jax 0.7.0). The runtime has run at ``xla_gpu_autotune_level=0``
since 969d56431, so a divergent hit/miss pattern now costs the missing rank
one compile and nothing waits. Measured at P4 (runs/DEV/771, agreement
off on a warm namespace): completes, eqp bitwise, hits on every rank.

DRIVERS MUST NOT CALL :func:`ensure_jax_compile_cache`: ``runtime.
initialize_communicator_stack`` step 7 owns it, above every jit in the
process. The two non-driver callers (``gw.w_isdf`` and ``gw.ppm_tau_kernel``
kernels imported standalone by tests) are idempotent re-entries.
"""
from __future__ import annotations

import atexit
import functools
import hashlib
import json
import os
import re
import sys
import threading
import time
import uuid
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path

_COMPILATION_CACHE_READY = False

# jax/_src/lru_cache.py
_CACHE_SUFFIX = "-cache"

_KV_NS = "lorrax/compile_cache/v1"

# Per-module compile fingerprints use a separate protocol.  Unlike the
# startup cache snapshot above, this one executes before EVERY backend
# compile and refuses a rank-divergent module before XLA can enter collective
# GPU autotuning and wait forever.
_COMPILE_KV_NS = "lorrax/compile_agreement/v2"
# Production Si MPA legitimately reaches the post-planning host-materialize
# compile slot 122 s apart across P4 ranks (JID 57909046.129).  Five minutes
# keeps that measured skew inside the contract while remaining a finite
# fail-fast bound; tests and bisect probes use their own short deadline.
# Default 0: wait without bound.  A late rank is skew, not disagreement.  The
# Sigma rule planner fits its windows round-robin across ranks, and on Si
# b80/c504 one rank's share ran past the former 300-second deadline (JID
# 57927048.48): rank 0 refused, tore down, and hung in a collective H5Fclose
# while its peers were still fitting.  A deadline turned skew into a hang.
_COMPILE_AGREEMENT_TIMEOUT_DEFAULT_S = 0.0


def _deadline_text(timeout_s: float) -> str:
    return (f"a {timeout_s:g}-second deadline" if timeout_s > 0
            else "no deadline (rank 0 names a missing rank every 60 s)")


def _agreement_deadline_env(name: str, default: float) -> float:
    """Read a finite deadline in seconds; 0 (the default) waits without bound."""
    raw = os.environ.get(name)
    try:
        value = default if raw is None or not raw.strip() else float(raw)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"{name} must be a number of seconds (0 = no deadline), "
            f"got {raw!r}") from exc
    if not (value >= 0.0 and value < float("inf")):
        raise ValueError(
            f"{name} must be a finite non-negative number of seconds, "
            f"got {raw!r}")
    return value


def _timeout_ms(timeout_s: float) -> int:
    """A deadline in seconds as the control plane's milliseconds (0 = none)."""
    return max(1, int(round(timeout_s * 1000))) if timeout_s > 0 else 0

# Parallel page-cache prefetch of the agreed entries (see _prefetch_agreed).
# ON: at 606 centroids / P=16 the SERIAL reads of 169 entries cost 29 s on one
# rank and 8.8 s on another, against the ~4.5 s of XLA compile they replace —
# i.e. without this the cache is a net LOSS on a cold-read CPU run.  876 kB of
# payload, so it is pure per-file Lustre latency under 16-way concurrency.


class _CacheState:
    """Per-process counters: the receipt (:func:`compile_cache_stats`)."""

    def __init__(self) -> None:
        self._compile_event_lock = threading.RLock()
        # Per thread: ``current`` = (module, key, request number) of the request
        # this thread is compiling; ``slot`` = the request number compile_ahead
        # reserved for the program this helper thread compiles.
        self._compile_local = threading.local()
        self._inflight: dict = {}      # stable MLIR key -> Future of its executable
        self._pool = None              # compile_ahead's helper threads
        self.enabled = False
        self.dir = ""
        self.n_proc = 1
        self.proc_idx = 0
        self.probes = 0        # persistent-cache lookups JAX asked for
        self.hits = 0          # lookups served from disk
        self.compiles = 0      # actual XLA compiles (backend_compile_and_load)
        self.compile_secs = 0.0
        # Compiles of modules with host callbacks, which JAX's persistent
        # cache never writes (compiler._cache_write): name -> [count, secs].
        self.uncacheable: dict = {}
        self.receipt_mark = (0, 0.0, 0, {})   # compile_receipt's last window edge
        self.read_secs = 0.0   # time spent loading executables from disk
        self.namespace = ""    # default-policy namespace ("" when explicit/off)
        self.compile_agreement_configured = False
        self.compile_agreement_enabled = False
        self.compile_agreement_reason = "not installed"
        self.compile_agreement_timeout_s = 0.0
        self.compile_agreement_checks = 0
        self.compile_fingerprint_secs = 0.0
        self.compile_agreement_secs = 0.0
        self._compile_client = None
        self._compile_sequence = 0
        self.probe_keys: set[str] = set()


_STATE = _CacheState()

class UnsafeCachePolicy(RuntimeError):
    """A requested disk-cache lifecycle would violate the cache contract."""


class CompileAgreementError(RuntimeError):
    """The ranks did not present the same module to the compile boundary."""


def _cache_size_policy(n_proc: int, max_size: int) -> bool:
    """Whether the persistent cache may run under JAX's size policy.

    JAX's ``0`` is an explicit cache-off request. A positive limit enables
    JAX's live LRU eviction, whose ``get`` rewrites an atime file under a
    file lock on every hit on every rank: at P64 and a thousand programs
    that is a Lustre metadata storm, so it is refused at P > 1 (rank 0's
    age pruner, :func:`_prune_namespaces`, bounds the default location).
    Returns ``False`` only for the standard cache-off spelling.
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
            "JAX_COMPILATION_CACHE_MAX_SIZE enables JAX's live LRU eviction, "
            f"which rewrites an atime file on every hit on every rank: unsafe "
            f"on a shared filesystem at P={n_proc}. Leave it unset (the "
            "default location is pruned by age) or use ISDF_JAX_CACHE_DIR=\"\" "
            "for a one-shot run.")
    return True

def _say(msg: str) -> None:
    # stderr: a driver's production stdout sinks incidental prints, and these
    # lines are the ones meant to stay loud in production.
    print(f"  [compile-cache] {msg}", file=sys.stderr, flush=True)


def _debug_say(msg: str) -> None:
    """Healthy cache telemetry follows the driver's one debug switch."""
    try:
        from runtime import debug_print_enabled
        enabled = debug_print_enabled()
    except Exception:                                      # noqa: BLE001
        enabled = False
    if enabled:
        _say(msg)


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


# the shard-slice patch: one ``jit__multi_slice`` program for every rank
# ---------------------------------------------------------------------------
# The cross-rank compile agreement refuses a program that differs between
# ranks. JAX's own ``ArrayImpl._multi_slice`` is one: ``shard_device_array``
# bakes THIS rank's shard offsets into a ``jit(static_argnums=...)``
# signature (507e16eda). This makes it one program on every rank: shard
# sizes static, offsets dynamic.
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
        _say("shard-slice-absent: " f"jax {_jax_generation()} has no ArrayImpl._multi_slice; the "
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
            _say("shard-slice-fallback: " f"the canonical shard slicer declined "
                    f"({type(exc).__name__}: {exc}); falling back to JAX's "
                    f"rank-dependent ArrayImpl._multi_slice.  Expect one "
                    f"jit__multi_slice cache key PER RANK.")
            return orig(self, start_indices, limit_indices, removed_dims)

    ArrayImpl._multi_slice = _canonical_multi_slice
    ArrayImpl._lorrax_shard_slice_orig = orig
    ArrayImpl._lorrax_shard_slice_installed = True


# ---------------------------------------------------------------------------
# the one jax._src lookup this file observes (counts only; JAX decides)
# ---------------------------------------------------------------------------
def _install_observation_patch() -> None:
    """Count persistent-cache probes and hits; rank 0 touches what it hits.

    ``get_executable_and_time`` is ``(cache_key, compile_options, backend,
    executable_devices)`` on the supported 0.9.1 wheel (``runtime.jax_support``
    asserts the arity at startup). ``*passthrough`` forwards the rest
    untouched; ``wraps`` keeps ``__wrapped__`` for that startup check.
    """
    from jax._src import compilation_cache as _cc

    if getattr(_cc, "_lorrax_observer_installed", False):
        return
    _orig_get = _cc.get_executable_and_time

    @functools.wraps(_orig_get)
    def _observed_get(cache_key, *passthrough):
        _STATE.probes += 1
        _STATE.probe_keys.add(cache_key)
        t0 = time.monotonic()
        try:
            executable, compile_time = _orig_get(cache_key, *passthrough)
        finally:
            _STATE.read_secs += time.monotonic() - t0
        if executable is not None:
            _STATE.hits += 1
            if _STATE.proc_idx == 0 and _STATE.namespace:
                _touch((cache_key,))
        return executable, compile_time

    _cc.get_executable_and_time = _observed_get
    _cc._lorrax_observer_installed = True


_COMPILE_ENTRY_POINT = "backend_compile_and_load"
#: The ``jax._src.compiler`` entry point every compile request goes through,
#: before the persistent-cache lookup.
_COMPILE_REQUEST_POINT = "compile_or_get_cached"


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
        mlir = operation.get_asm(binary=False, enable_debug_info=False)
    except Exception as exc:                               # noqa: BLE001
        raise CompileAgreementError(
            "GATE cross_rank_compile_agreement: REFUSED before compiling "
            f"module {name!r}: its stable MLIR fingerprint could not be "
            f"computed ({type(exc).__name__}: {exc}).") from exc
    if isinstance(mlir, str):
        mlir = mlir.encode("utf-8")
    digest = hashlib.sha256(canonical_compile_text(bytes(mlir))).hexdigest()
    return name, digest, time.monotonic() - t0


#: Process-local integer attributes that a rank-symmetric program may carry
#: verbatim in its MLIR: the cuBLASMp/cuSolverMp context pointer is emitted as
#: ``ctx_handle = 93965789305168 : i64`` on the distributed GEMM custom call,
#: and it differs on every rank by construction.  Measured 2026-09-04 (claim
#: 704): the Na metal chi0 route lowered the identical GEMM plan on four ranks
#: with four different handles, and the exact-bytes agreement refused it.
#: Replacing only the handle made all four SHA-256 digests identical.
_HANDLE_ATTR_RE = re.compile(rb"\b([A-Za-z_]*handle)\s*=\s*-?\d+")


def canonical_compile_text(mlir: bytes) -> bytes:
    """Return ``mlir`` with process-local handle literals replaced.

    Only attributes whose name ends in ``handle`` are touched, and only
    their integer literal; shapes, layouts, constants and every other
    attribute stay byte-exact, so a genuinely rank-divergent program still
    disagrees.  Pure so the test can pin it without a device.
    """
    return _HANDLE_ATTR_RE.sub(rb"\1=<process-local>", mlir)


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


def _format_compile_refusal(module_name, occurrence, reason, records) -> str:
    rank_lines = []
    for rank, record in enumerate(records):
        if record is None:
            rank_lines.append(f"rank {rank}: <not-arrived>")
        else:
            rank_lines.append(
                f"rank {rank}: key={record.get('key', '<missing>')} "
                f"module={record.get('module', '<missing>')!r}")
    return (
        "GATE cross_rank_compile_agreement: REFUSED before XLA compilation.\n"
        f"  got: {reason}; module {module_name!r}, compile request "
        f"{occurrence}.\n"
        f"  rank keys: {'; '.join(rank_lines)}.\n"
        "  want: every rank to present the same stable MLIR/HLO key for the "
        "same compile request.\n"
        "  why: a rank-divergent program hangs silently in its next "
        "collective (INVARIANTS 21).\n"
        "  fix: remove rank-conditional shapes/jits or make the emitted "
        "module identical; LORRAX_JAX_COMPILE_AGREEMENT=0 is an UNSAFE "
        "bisect-only opt-out.")


def _publish_compile_record(module_name: str, key: str, occurrence: int) -> None:
    """Every rank, on every compile request (a cache hit included): its fingerprint."""
    from ffi.common.broadcast import publish_rank_record
    s = _STATE
    record = {"rank": s.proc_idx, "module": module_name,
              "occurrence": occurrence, "key": key}
    publish_rank_record(s._compile_client, _compile_event_prefix(occurrence),
                        s.proc_idx, s.n_proc,
                        json.dumps(record, sort_keys=True).encode("utf-8"))


def _check_compile_record(module_name: str, key: str, occurrence: int) -> None:
    """A rank about to compile: refuse by name if any rank's record differs.

    Rank 0 reads every rank's record for this request (they all publish,
    whether they hit or compile, so nobody is waited on for a compile it
    never does); a peer reads rank 0's. A rank that hits the cache checks
    nothing and waits for nothing, so an asymmetric hit costs it nothing
    and the compiling ranks one exchange. With no deadline (the default)
    the wait names a missing rank every 60 s.
    """
    from ffi.common.broadcast import collect_rank_records, wait_for_key
    s = _STATE
    prefix = _compile_event_prefix(occurrence)
    timeout_s = s.compile_agreement_timeout_s
    t0 = time.monotonic()
    reason = ""
    try:
        if s.proc_idx == 0:
            payloads = collect_rank_records(
                s._compile_client, prefix, s.n_proc, _timeout_ms(timeout_s),
                what=f"every rank's record of the {module_name} compile request")
            records = [_decode_compile_record(payload, rank)
                       for rank, payload in enumerate(payloads)]
        else:
            # An early peer may wait nearly a whole deadline for rank 0, so a
            # finite deadline is doubled with a handoff allowance.
            peer_wait_s = 2.0 * timeout_s + min(2.0, max(0.1, timeout_s * 0.1)) if timeout_s > 0 else 0.0
            payload = wait_for_key(
                s._compile_client, f"{prefix}/rank/0", _timeout_ms(peer_wait_s),
                what=f"rank 0's record of the {module_name} compile request")
            records = [_decode_compile_record(payload, 0)]
    except CompileAgreementError:
        raise
    except Exception as exc:                               # noqa: BLE001
        records, reason = [], (f"a rank's record did not arrive within the "
                               f"deadline ({type(exc).__name__}: {exc})")
    s.compile_agreement_checks += 1
    s.compile_agreement_secs += time.monotonic() - t0
    if not reason and any(r.get("key") != key or r.get("module") != module_name
                          for r in records):
        reason = "ranks published different stable MLIR/HLO keys"
    if reason:
        raise CompileAgreementError(
            _format_compile_refusal(module_name, occurrence, reason, records))


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


    # One ``jit__multi_slice`` program on every rank before the mesh warm-up
    # compiles it; otherwise the agreement below refuses JAX's own slicer.
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

    timeout_s = _agreement_deadline_env(
        "LORRAX_JAX_COMPILE_AGREE_TIMEOUT_S",
        _COMPILE_AGREEMENT_TIMEOUT_DEFAULT_S)
    s._compile_client = client
    s.compile_agreement_timeout_s = timeout_s
    s.compile_agreement_enabled = True
    s.compile_agreement_reason = "enabled"
    if s.proc_idx == 0:
        _debug_say(
            "cross-rank compile agreement enabled before backend compile "
            f"with {_deadline_text(timeout_s)}.")


def install_compile_agreement() -> None:
    """Install the compile counter and default-on all-rank module refusal.

    The runtime calls this after ``jax.distributed.initialize`` and before
    mesh warmup.  Direct library/test paths are safe: P=1 or an absent
    coordination client makes agreement an announced no-op, while the
    compile counter remains useful.
    """
    _configure_compile_agreement()
    _install_compile_counter()
    _install_device_fit_gate()


def _install_device_fit_gate() -> None:
    """Refuse a module larger than the device before its first execution.

    GATE xla_rematerialization (``runtime.aot_memory.refuse_over_device``).
    XLA's rematerialization pass is off (``runtime.disable_xla_rematerialization``),
    so nothing shrinks a module that does not fit; it would fail at its first
    allocation.  The check runs when jax builds the module's executor
    (``pxla.ExecuteReplicated``), which it does lazily on the first call of a
    compiled module, on every path (jit dispatch and AOT ``Compiled``), before
    any of the module's buffers exist.  It does not run at compile: planners
    compile larger candidates on purpose to read their figures and then
    shrink (``runtime.aot_memory.check_chunk``), and a candidate that never
    runs must not refuse.

    The wrap is signature-checked at install, like the other ``jax._src``
    patches here: when ``ExecuteReplicated`` is absent or its ``__init__`` does
    not start ``(self, xla_executable, name, backend)``, the gate is skipped
    with one stderr notice.  A check that cannot read a module's figures lets
    it run, with the same notice; only :class:`ModuleDoesNotFit` propagates.
    """
    import inspect
    from jax._src.interpreters import pxla
    from runtime.aot_memory import ModuleDoesNotFit, refuse_over_device

    if getattr(pxla, "_lorrax_device_fit_gate_installed", False):
        return
    cls = getattr(pxla, "ExecuteReplicated", None)
    head = ("self", "xla_executable", "name", "backend")
    try:
        params = tuple(inspect.signature(cls.__init__).parameters)[:4]
    except Exception:                                      # noqa: BLE001
        params = None
    if cls is None or params != head:
        # Same discipline as the other jax._src patches: a changed private
        # surface is skipped with one notice, never mis-wrapped.
        _fit_gate_notice(
            f"jax {_jax_generation()}: pxla.ExecuteReplicated.__init__ is "
            f"{'absent' if cls is None else f'{params!r}'}, not {head!r}; "
            f"GATE xla_rematerialization is NOT armed (a module larger than "
            f"the device fails at allocation instead).")
        return
    _orig_init = cls.__init__

    @functools.wraps(_orig_init)
    def _gated_init(self, xla_executable, name, backend, *args, **kwargs):
        try:
            refuse_over_device(xla_executable, str(name),
                               str(getattr(backend, "platform", "")))
        except ModuleDoesNotFit:
            raise
        except Exception as exc:                           # noqa: BLE001
            _fit_gate_notice(
                f"the device-fit check could not read module {name!r} "
                f"({type(exc).__name__}: {exc}); it runs unchecked.")
        _orig_init(self, xla_executable, name, backend, *args, **kwargs)

    cls.__init__ = _gated_init
    pxla._lorrax_device_fit_gate_installed = True


_FIT_GATE_SAID = False


def _fit_gate_notice(msg: str) -> None:
    """One stderr line per process, rank 0 only, when the fit gate stands down."""
    global _FIT_GATE_SAID
    if _FIT_GATE_SAID:
        return
    _FIT_GATE_SAID = True
    if _STATE.proc_idx == 0:
        print(f"  [compile-cache] jax-compat: {msg}", file=sys.stderr,
              flush=True)


def _install_compile_counter() -> None:
    """Count real XLA compiles and run the compile agreement around them.

    ``compile_or_get_cached`` is every compile request (the persistent-cache
    lookup included) and ``backend_compile_and_load`` every real compile;
    both are module attributes of ``jax._src.compiler`` on the supported
    0.9.1 wheel. A request publishes this rank's fingerprint under its
    global request number; a real compile checks the other ranks' records
    (:func:`_check_compile_record`). The request number is taken under one
    lock in request order on the main thread, or is the slot
    :func:`compile_ahead` reserved there for a helper thread, so request
    numbers agree across ranks that request the same programs in the same
    order (INVARIANTS 21, 25). The lock does not cover the backend compile:
    helper threads compile beside the main thread. A request for a module
    another thread is compiling waits for that executable (one compile per
    module per process).

    Raises :class:`_JaxSurfaceUnsupported` when an entry point is absent, so
    the caller reports the storm telemetry OFF instead of a confident 0.
    """
    from jax._src import compiler as _compiler

    if getattr(_compiler, "_lorrax_compile_counter_installed", False):
        return
    for name in (_COMPILE_ENTRY_POINT, _COMPILE_REQUEST_POINT):
        if getattr(_compiler, name, None) is None:
            raise _JaxSurfaceUnsupported(
                f"jax._src.compiler has no {name} on this jax "
                f"({_jax_generation()}): no entry point to count real XLA "
                f"compiles at (jax 0.5.3 spelled it backend_compile; that "
                f"line was dropped with the move to jax 0.7.0).")
    _orig = getattr(_compiler, _COMPILE_ENTRY_POINT)
    _orig_request = getattr(_compiler, _COMPILE_REQUEST_POINT)

    def _requested(*args, **kwargs):
        s = _STATE
        if not (s.compile_agreement_enabled or s._pool is not None):
            return _orig_request(*args, **kwargs)
        module = args[1] if len(args) > 1 else kwargs.get("computation")
        module_name, key, fingerprint_secs = _compile_module_identity(module)
        local = s._compile_local
        with s._compile_event_lock:
            s.compile_fingerprint_secs += fingerprint_secs
            occurrence = getattr(local, "slot", None)
            if occurrence is None:
                occurrence = s._compile_sequence
                s._compile_sequence += 1
            if s.compile_agreement_enabled:
                _publish_compile_record(module_name, key, occurrence)
            future = s._inflight.get(key)
            owner = future is None
            if owner:
                future = s._inflight[key] = Future()
        if not owner:
            return future.result()
        local.current = (module_name, key, occurrence)
        try:
            executable = _orig_request(*args, **kwargs)
        except BaseException as exc:
            future.set_exception(exc)
            raise
        finally:
            local.current = None
            with s._compile_event_lock:
                s._inflight.pop(key, None)
        future.set_result(executable)
        return executable

    def _uncacheable(module, host_callbacks, seconds):
        # backend_compile_and_load(backend, module, devices, options, host_callbacks)
        if not host_callbacks:
            return
        try:
            attr = module.operation.attributes["sym_name"]
            module_name = str(getattr(attr, "value", attr)).strip('"')
        except Exception:                                  # noqa: BLE001
            module_name = "<unnamed-module>"
        row = _STATE.uncacheable.setdefault(module_name, [0, 0.0])
        row[0] += 1
        row[1] += seconds
        if row[0] == 1 and _STATE.proc_idx == 0:
            _say(f"UNCACHEABLE {module_name}: {len(host_callbacks)} host callbacks "
                 f"({seconds:.1f} s); JAX's persistent cache never stores it, so it "
                 f"compiles again in every process")

    def _counting(*args, **kwargs):
        current = getattr(_STATE._compile_local, "current", None)
        if _STATE.compile_agreement_enabled and current is not None:
            _check_compile_record(*current)
        module = args[1] if len(args) > 1 else kwargs.get("module")
        callbacks = args[4] if len(args) > 4 else kwargs.get("host_callbacks")
        t0 = time.monotonic()
        try:
            return _orig(*args, **kwargs)
        finally:
            with _STATE._compile_event_lock:
                _STATE.compiles += 1
                _STATE.compile_secs += time.monotonic() - t0
                _uncacheable(module, callbacks, time.monotonic() - t0)

    setattr(_compiler, _COMPILE_REQUEST_POINT, _requested)
    setattr(_compiler, _COMPILE_ENTRY_POINT, _counting)
    _compiler._lorrax_compile_counter_installed = True


#: Helper threads per process for :func:`compile_ahead`, at most.  Past eight
#: the node's cores are full with four ranks compiling, each compile stretches
#: and a live call waiting for ITS program pays that stretch; eight keeps the
#: scaling and half the host memory of sixteen concurrent XLA compiles
#: (sandbox claim 4178).
COMPILE_THREADS_MAX = 8


def _pool() -> ThreadPoolExecutor:
    s = _STATE
    if s._pool is None:
        try:
            from runtime import _physical_cores_in_affinity
            cores = _physical_cores_in_affinity() or 1
        except Exception:                                  # noqa: BLE001
            cores = max(1, len(os.sched_getaffinity(0)) // 2)
        s._pool = ThreadPoolExecutor(max_workers=max(1, min(COMPILE_THREADS_MAX, cores)),
                                     thread_name_prefix="lorrax-compile")
    return s._pool


def _compile_in_slot(lowered, slot):
    local = _STATE._compile_local
    local.slot = slot
    try:
        return lowered.compile()
    finally:
        local.slot = None


def compile_ahead(program, *args) -> Future:
    """Compile ``program.lower(*args)`` on a helper thread; return the Future of its ``Compiled``.

    For a stage owner that knows a program's avals before its first call
    (``program`` a ``jax.jit``, ``args`` arrays or ``jax.ShapeDtypeStruct``
    with the live call's shardings). The lowering runs HERE, on the calling
    thread in program order, so every rank lowers the same module at the same
    request number, and that number is reserved now for the helper (INVARIANTS
    21, 25); only XLA's backend compile, which releases the GIL, leaves the
    thread. The live call then finds the executable on its lowering
    (``MeshComputation.compile`` memoizes it) or waits for the compile in
    flight (:func:`_install_compile_counter`); it never compiles twice. A
    refusal raised by the compile (the agreement, a module over the device)
    surfaces at ``.result()`` or at the live call, on the calling thread.
    Helper threads never dispatch an executable.
    """
    lowered = program.lower(*args)
    s = _STATE
    with s._compile_event_lock:
        slot = s._compile_sequence
        s._compile_sequence += 1
    return _pool().submit(_compile_in_slot, lowered, slot)


def compile_threads() -> int:
    """Helper threads :func:`compile_ahead` uses in this process (0 until its first call)."""
    pool = _STATE._pool
    return 0 if pool is None else int(pool._max_workers)



def compile_receipt(label: str, emit=None) -> str:
    """Print this window's compile receipt on rank 0 and start the next window.

    ``<label> compile: real R (S s), cache hits H, uncacheable U [names]``:
    XLA compiles, their seconds, persistent-cache hits and compiles of modules
    JAX's cache cannot store (host callbacks), since the previous receipt.  The
    SC loop prints one per map and each driver one at its end, so a run that is
    killed still reports every finished map.  A map past the second with R > 0
    recompiled something it already had; U > 0 recompiles in every process.
    ``emit`` writes the line (the SC map's record), else :func:`_say`.
    """
    s = _STATE
    compiles, secs, hits, seen = s.receipt_mark
    names = sorted(n for n, row in s.uncacheable.items() if row[0] > seen.get(n, 0))
    uncacheable = sum(row[0] - seen.get(n, 0) for n, row in s.uncacheable.items())
    line = (f"{label} compile: real {s.compiles - compiles} ({s.compile_secs - secs:.1f} s), "
            f"cache hits {s.hits - hits}, uncacheable {uncacheable}"
            + (f" [{', '.join(names)}]" if names else ""))
    s.receipt_mark = (s.compiles, s.compile_secs, s.hits,
                      {n: row[0] for n, row in s.uncacheable.items()})
    if s.proc_idx == 0:
        (emit or _say)(line)
    return line


def _report() -> None:
    try:
        _report_impl()
    except Exception:                                      # noqa: BLE001
        pass


def _report_impl() -> None:
    s = _STATE
    _debug_say(f"rank {s.proc_idx}/{s.n_proc} summary: "
               f"xla_compiles={s.compiles} ({s.compile_secs:.2f}s)  "
               f"compile_agreement={s.compile_agreement_checks} "
               f"({s.compile_fingerprint_secs:.3f}s fingerprint + "
               f"{s.compile_agreement_secs:.3f}s exchange; "
               f"{s.compile_agreement_reason})  "
               f"cache_probes={s.probes} hits={s.hits} ({s.read_secs:.2f}s)  "
               f"enabled={s.enabled} dir={s.dir}")


def compile_cache_stats() -> dict:
    """Snapshot this rank's cache counters."""
    s = _STATE
    return {
        "enabled": s.enabled, "dir": s.dir, "n_proc": s.n_proc,
        "proc_idx": s.proc_idx, "probes": s.probes, "hits": s.hits,
        "compiles": s.compiles, "compile_secs": s.compile_secs,
        "compile_agreement_configured": s.compile_agreement_configured,
        "compile_agreement_enabled": s.compile_agreement_enabled,
        "compile_agreement_reason": s.compile_agreement_reason,
        "compile_agreement_timeout_s": s.compile_agreement_timeout_s,
        "compile_agreement_checks": s.compile_agreement_checks,
        "compile_fingerprint_secs": s.compile_fingerprint_secs,
        "compile_agreement_secs": s.compile_agreement_secs,
        "compile_threads": compile_threads(),
        "read_secs": s.read_secs, "namespace": s.namespace,
        "is_cache_writer": s.proc_idx == 0,
        "write_scope": "process-local; JAX writes on process 0 only",
        "keys": sorted(s.probe_keys),
    }


# the default location: one namespace per jax/jaxlib/FFI bundle, pruned by rank 0
# ---------------------------------------------------------------------------
#: Retention.  Nothing used in the last five days is removed: the longest job
#: wall on either machine is 120 h (Frontera's long queue; Perlmutter's is
#: 48 h), so a live job's entries stay in place until the job exits and
#: its peers can still read them.  Rank 0
#: touches what it uses, so an mtime is a last use.  An entry goes after a week
#: unused; another namespace too, or earlier, least recently used first, while
#: the tree exceeds either cap.  Measured (P4, 3697ea6e): MoS2 bispinor 606
#: entries / 4.0 MB, Fe 4^3 bispinor 1261 / 18 MB.
_NS_LIVE_S = 5 * 86400
_NS_TTL_S = 7 * 86400
_NS_MAX_BYTES = 2 << 30
_NS_MAX_FILES = 200_000
_NS_PRUNE_EVERY_S = 6 * 3600
_NS_STAMP = ".last_used"
_NS_PRUNE_STAMP = ".last_prune"
#: Bump when this file changes how a key is hashed (the invariant-key and
#: shard-slice patches) or how an entry is stored (the atomic writer): those
#: change what an entry means without changing jax, jaxlib or the FFI bundle.
_KEY_SCHEMA = "k1"


def default_cache_root() -> Path:
    """``$SCRATCH/.cache/lorrax/jax_compile`` (``~`` where there is no SCRATCH).

    Beside the k-convolution cubin cache (``ffi.fft.cubin_cache_dir``): on
    scratch, never home, and one tree per user.
    """
    from lxkit import user_cache_dir
    return user_cache_dir("jax_compile")


def _library_identity(path) -> str:
    """A native library's sealed bundle id, or its own SHA-256 when unsealed."""
    p = Path(path).resolve()
    for manifest in (p.parent / "lorrax_ffi_bundle.json",
                     p.parent.parent / "lorrax_ffi_bundle.json"):
        if manifest.is_file():
            bundle = json.loads(manifest.read_text(encoding="utf-8"))
            return "bundle-" + str(bundle["bundle_id"])[:12]
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return "so-" + h.hexdigest()[:12]


def _ffi_identity() -> str:
    """The FFI libraries this process loaded, or would load first."""
    try:
        from ffi.common import ffi_loader
    except Exception:                                      # noqa: BLE001
        return "noffi"
    ids = set()
    for platform in ("CUDA", "cpu"):
        path = ffi_loader.loaded_lib_path(platform)
        if path is None:
            path = next((c for c in ffi_loader._candidate_paths(platform)
                         if c.is_file()), None)
        if path is not None:
            try:
                ids.add(_library_identity(path))
            except (OSError, ValueError, KeyError):
                ids.add("unreadable")
    return "+".join(sorted(ids)) or "noffi"


def cache_namespace() -> str:
    """``jax<v>-jaxlib<v>_<ffi>_<schema>``: what JAX's own key cannot see.

    JAX keys an entry on the module, the compile options, the jaxlib version
    and the backend.  It does not see the native bundle behind a custom-call
    name (a handler's traits, e.g. command-buffer compatibility, are read at
    compile time).  The LORRAX source is deliberately absent: a commit reaches
    a compiled program only through its HLO, which the key already hashes.
    """
    import jax
    from jax._src.lib import version_str as jaxlib_version
    raw = (f"jax{jax.__version__}-jaxlib{jaxlib_version}_{_ffi_identity()}"
           f"_{_KEY_SCHEMA}")
    return re.sub(r"[^A-Za-z0-9._+-]", "-", raw)


def default_cache_dir() -> Path:
    """The default-policy base: :func:`default_cache_root` / namespace."""
    return default_cache_root() / cache_namespace()


def _prune_namespaces(root: Path, current: str, *, keep=frozenset(),
                      now: float | None = None) -> list[str]:
    """Retire stale entries and namespaces under ``root``; return what went.

    In ``current``, an entry untouched for :data:`_NS_TTL_S` and not in ``keep``
    goes.  Another namespace is kept, newest-used first, while used within
    :data:`_NS_LIVE_S`, or both younger than :data:`_NS_TTL_S` and inside the
    caps counted so far.  Removal is a rename into ``root/.trash`` (atomic, so
    no run lists a half-deleted one) and then a best-effort delete.
    """
    import shutil

    now = time.time() if now is None else float(now)
    trash = root / ".trash"
    usage, stale = [], []
    for entry in os.scandir(root):
        if entry.name.startswith(".") or not entry.is_dir(follow_symlinks=False):
            continue
        try:
            used = os.stat(os.path.join(entry.path, _NS_STAMP)).st_mtime
        except OSError:
            used = entry.stat(follow_symlinks=False).st_mtime
        nbytes = nfiles = 0
        for dirpath, _dirs, files in os.walk(entry.path):
            nfiles += len(files)
            for name in files:
                try:
                    st = os.stat(os.path.join(dirpath, name))
                except OSError:
                    continue
                nbytes += st.st_size
                if (entry.name == current and name.endswith(_CACHE_SUFFIX)
                        and now - st.st_mtime > _NS_TTL_S
                        and name[: -len(_CACHE_SUFFIX)] not in keep):
                    stale.append(os.path.join(dirpath, name))
        usage.append((used, entry.name, nbytes, nfiles))
    usage.sort(reverse=True)
    removed: list[str] = []
    total_bytes = total_files = 0
    for used, name, nbytes, nfiles in usage:
        total_bytes += nbytes
        total_files += nfiles
        age = now - used
        if name == current or age < _NS_LIVE_S:
            continue
        if (age > _NS_TTL_S or total_bytes > _NS_MAX_BYTES
                or total_files > _NS_MAX_FILES):
            try:
                trash.mkdir(exist_ok=True)
                os.rename(root / name, trash / f"{name}.{uuid.uuid4().hex[:8]}")
            except OSError:
                continue
            removed.append(name)
            total_bytes -= nbytes
            total_files -= nfiles
    gone = 0
    for path in stale:
        try:
            trash.mkdir(exist_ok=True)
            os.rename(path, trash / f"entry.{uuid.uuid4().hex}")
            gone += 1
        except OSError:
            pass
    if gone:
        removed.append(f"{gone} entries of {current}")
    shutil.rmtree(trash, ignore_errors=True)
    return removed


def _touch(keys) -> None:
    """Mark these entries of the bound directory used now (rank 0 only)."""
    for key in keys:
        try:
            os.utime(os.path.join(_STATE.dir, key + _CACHE_SUFFIX))
        except OSError:
            pass


def _start_namespace_prune(root: Path, current: str, keep=frozenset()) -> None:
    """Rank 0, in a daemon thread: touch ``keep`` (the agreed set), then prune.

    The touch runs every time, the prune at most every :data:`_NS_PRUNE_EVERY_S`,
    off the startup path (the walk stats every file).  A process that exits
    mid-walk leaves only atomic renames; the next prune finishes the delete.
    """
    def _run() -> None:
        try:
            _touch(keep)
            stamp = root / _NS_PRUNE_STAMP
            if (stamp.exists() and time.time() - stamp.stat().st_mtime
                    < _NS_PRUNE_EVERY_S):
                return
            stamp.touch()
            removed = _prune_namespaces(root, current, keep=keep)
            if removed:
                _debug_say(f"pruned under {root}: {', '.join(removed)}")
        except Exception as exc:                           # noqa: BLE001
            _say(f"namespace prune under {root} failed "
                 f"({type(exc).__name__}: {exc}); nothing is lost but space.")

    threading.Thread(target=_run, name="lorrax-jax-cache-prune",
                     daemon=True).start()


def _resolve_cache_base_dir() -> tuple[str, str]:
    """Resolve the one persistent-cache owner: ``(base, source)``.

    A nonempty ``ISDF_JAX_CACHE_DIR`` is used as-is (``"explicit"``); an
    empty or whitespace one is the explicit opt-out.  Unset, the runtime
    default is :func:`default_cache_dir` (``"runtime default"``).  No other
    variable (``LORRAX_RUN_DIR``, ``XDG_CACHE_HOME``, JAX's own
    ``JAX_COMPILATION_CACHE_DIR``) selects a location.
    """
    explicit = os.environ.get("ISDF_JAX_CACHE_DIR")
    if explicit is not None:
        return explicit.strip(), "explicit"
    return str(default_cache_dir()), "runtime default"


# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# public entry point
# ---------------------------------------------------------------------------
def ensure_jax_compile_cache() -> None:
    """Arm JAX's persistent compile cache once per process (see the module docstring).

    ONE caller owns this: ``runtime.initialize_communicator_stack`` step 7.
    Idempotent, so a stray re-entry through an import order is harmless
    rather than a second, differently configured cache.
    """
    global _COMPILATION_CACHE_READY
    if _COMPILATION_CACHE_READY:
        return
    _COMPILATION_CACHE_READY = True

    import jax as _jax
    try:
        n_proc = _jax.process_count()
        proc_idx = _jax.process_index()
    except Exception:                                      # noqa: BLE001
        n_proc, proc_idx = 1, 0
    _STATE.n_proc, _STATE.proc_idx = n_proc, proc_idx

    # The compile counter goes in on EVERY path, cache-off included, so
    # "compiles with the cache" and "compiles without it" are one measurement.
    try:
        install_compile_agreement()
        atexit.register(_report)
    except Exception as exc:                               # noqa: BLE001
        if proc_idx == 0:
            _say(f"compile-storm telemetry OFF: the XLA compile counter did "
                 f"not install ({type(exc).__name__}: {exc}); xla_compiles "
                 f"will read 0 whatever this run compiles.")

    try:
        cache_dir, cache_source = _resolve_cache_base_dir()
    except Exception as exc:                               # noqa: BLE001
        cache_dir, cache_source = "", (
            f"the runtime-default namespace could not be resolved "
            f"({type(exc).__name__}: {exc})")
    if n_proc > 1 or not cache_dir:
        # JAX would otherwise auto-enable XLA's per-fusion caches, UPDATE on
        # process 0 and READ on its peers, with no real base directory.
        _jax.config.update("jax_persistent_cache_enable_xla_caches", "")
    if not cache_dir:
        if proc_idx == 0:
            reason = ("ISDF_JAX_CACHE_DIR=\"\" opt-out"
                      if cache_source == "explicit" else cache_source)
            _say(f"persistent compile cache OFF ({reason}). JAX's in-process "
                 f"executable cache remains active.")
        return

    from jax._src import config as _jax_config
    if not _cache_size_policy(n_proc, int(_jax_config.compilation_cache_max_size.value)):
        if proc_idx == 0:
            _say("persistent compile cache OFF (JAX_COMPILATION_CACHE_MAX_SIZE=0).")
        return
    # ONE directory per world size: entries are compiled for P devices.
    cache_path = Path(cache_dir).expanduser() / f"np{n_proc}"
    _STATE.dir = str(cache_path)
    try:
        cache_path.mkdir(parents=True, exist_ok=True)
        _jax.config.update("jax_compilation_cache_dir", str(cache_path))
        # Threshold 0 unless exported: JAX's 1 s default persisted 2 of 666
        # executables on the MoS2 bispinor deck.
        if os.environ.get("JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS") is None:
            _jax.config.update("jax_persistent_cache_min_compile_time_secs", 0.0)
        # JAX binds its cache object at the first compile that consults it
        # (the mesh warm-up runs before this step) and a later config update
        # does not rebind it; the reset makes the next compile read this dir.
        from jax._src import compilation_cache as _cc
        _cc.reset_cache()
    except Exception as exc:                               # noqa: BLE001
        if proc_idx == 0:
            _say(f"DISABLED: cannot arm {cache_path} ({exc}). Every rank "
                 f"compiles from scratch.")
        return
    _install_observation_patch()
    _STATE.enabled = True
    if cache_source != "explicit":
        _STATE.namespace = Path(cache_dir).name
        if proc_idx == 0:
            try:
                (Path(cache_dir) / _NS_STAMP).touch()
            except OSError as exc:
                _say(f"cannot stamp {cache_dir} ({exc}); it may be pruned early.")
            _start_namespace_prune(Path(cache_dir).parent, _STATE.namespace)
    if proc_idx == 0:
        _debug_say(f"ARMED at {n_proc} processes, {cache_path} "
                   f"({cache_source}; ISDF_JAX_CACHE_DIR overrides, \"\" opts out).")
