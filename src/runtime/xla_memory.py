"""How XLA's GPU memory pool is configured, and whether its numbers are real.

The **L3 (substrate)** half of the allocator story: a read-only mirror of
jaxlib's own parse of ``XLA_PYTHON_CLIENT_*`` / ``XLA_CLIENT_MEM_FRACTION``,
plus the corroboration of that environment against the live client, plus
:func:`pool_high_water`, the resettable device peak the stage receipt reads.  Nothing
here knows what a band or a q-point is; it belongs beside
:func:`runtime.set_default_env`, which is the module that decides which of
these variables LORRAX ships.

WHY IT MOVED HERE.  It lived in ``gw.gw_config`` — the GW driver's 2.5 kLoC
deck parser — and ``runtime.collect_startup_facts`` reached *up* into that
driver package to get it, through an import that was lazy and exception-
guarded precisely because the direction was wrong.  ``runtime/__init__.py``
carried the note "the two corroboration helpers live in ``gw.gw_config`` and
belong in ``runtime/xla_memory.py``, which is why this import is lazy,
guarded, and made only from the fact collector" (numbered request R9 of that
workstream).  This module is that request, landed.  ``gw.gw_config``
re-exports both names, so every existing import site keeps working.

IMPORTABLE WITHOUT JAX, like the module it came from: it reads
``os.environ`` and nothing else, so it is safe before backend init and
testable on a login node.  ``runtime.__init__`` is likewise jax-free at
module scope, so importing this costs nothing.
"""
from __future__ import annotations

import os
from dataclasses import dataclass

# The falsy vocabulary, taken from the runtime package rather than re-typed.
# Identical token set to the ``_ENV_FALSE`` this code used in ``gw_config``
# (same four spellings), and deliberately WITHOUT ``""`` — see the long note
# on ``runtime._FALSY_TOKENS``: blank means "unset", not "off".  The only
# consumer below is the preallocate-typo test, which has already rejected a
# blank value before it gets there, so the two are equivalent on every input
# that reaches it.
from . import _FALSY_TOKENS as _ENV_FALSE


# ---------------------------------------------------------------------------
#  XLA GPU memory environment — read-only mirror of jax's own parse
# ---------------------------------------------------------------------------
#
# Two call sites branch on these (``gw_init`` captions the ζ-fit peak,
# ``gw_output`` prints the banner), so a wrong reading is a wrong NUMBER in
# a report, not just cosmetics.  The authority is jaxlib's
# ``generate_pjrt_gpu_plugin_options`` — in this deployment
# ``.venv/lib/python3.12/site-packages/jaxlib/xla_client.py:181-222``:
#
#     allocator = os.getenv('XLA_PYTHON_CLIENT_ALLOCATOR', 'default').lower()
#     if allocator not in ('default', 'platform', 'bfc', 'cuda_async'): raise
#     memory_fraction            = os.getenv('XLA_CLIENT_MEM_FRACTION', '')
#     deprecated_memory_fraction = os.getenv('XLA_PYTHON_CLIENT_MEM_FRACTION', '')
#     # both set -> ValueError
#     preallocate = os.getenv('XLA_PYTHON_CLIENT_PREALLOCATE', '')
#     if preallocate: options['preallocate'] = preallocate not in ('false','False','0')
#
# FOUR traps encoded below, each of which previously produced a
# confidently wrong statement:
#
#  1. jax LOWERCASES the allocator; ``gw_init.py``'s ``== "platform"`` did
#     not, so ``=PLATFORM`` reported the peak as faithful when it was not.
#  2. ``platform``, ``cuda_async`` and BFC are THREE distinct allocators;
#     the old comment used the names interchangeably.  ``platform`` is
#     plain ``cudaMalloc``, NOT cudaMallocAsync.
#  3. ``config/frontera/ffi_env.sh:24`` deploys ``cuda_async``, which the
#     ``== "platform"`` test never matched.
#  4. jax's preallocate test is case-SENSITIVE, so
#     ``XLA_PYTHON_CLIENT_PREALLOCATE=FALSE`` leaves preallocation ON
#     while reading as "off" to a human.  ``unset`` also means ON (the
#     option is simply not passed and XLA preallocates by default).
#
# WHAT WAS MEASURED (allocator workstream, 8 GPUs / 2 nodes, job 7882447,
# each cell run twice with rep 2 in reverse order):
#
#   allocator     memory_stats()                      peak_bytes_in_use
#   ------------  ----------------------------------  ------------------
#   unset / bfc   fully populated                     1.000 / 6.500 GB
#   cuda_async    fully populated                     1.000 / 6.500 GB
#                                                     (IDENTICAL to BFC)
#   platform      bytes_limit=0, peak_bytes_in_use=0  0.000 GB  — BLIND
#
# So the premise the old branch was written on — "cuda_async returns freed
# transients to its pool, so the reading under-reports" — was NOT
# reproduced for steady allocations.  Transient-heavy kernels were not
# tested, so ``peak_note`` says that instead of claiming either way.  And
# ``platform`` is not "low", it is ZERO: any figure a run prints under
# ``platform`` came from the nvidia-smi fallback in
# ``isdf_fitting.fit_zeta_to_h5._track_peak``, which samples the WHOLE GPU
# (other processes included), not this run's arena.
#
# This function READS ONLY.  It never sets an allocator variable: which
# values LORRAX ships is decided in ``runtime.set_default_gpu_pool``
# (cuda_async, PREALLOCATE=true, fraction runtime.GPU_POOL_FRACTION on CUDA).  Everything here
# must stay correct under every one of them, including unset.

_XLA_ALLOCATORS = ("default", "platform", "bfc", "cuda_async")


def cuda_device_total_bytes(ordinal: int = 0) -> int | None:
    """The device's total memory as XLA sizes its pool from it (libcuda
    ``cuDeviceTotalMem``; creates no context).  None where there is no
    CUDA driver.  The startup report checks ``bytes_limit`` against
    ``fraction x`` this, because ``os.environ`` is a false witness once the
    client exists."""
    try:
        import ctypes
        cu = ctypes.CDLL("libcuda.so.1")
        dev, total = ctypes.c_int(), ctypes.c_size_t()
        if (cu.cuInit(0) or cu.cuDeviceGet(ctypes.byref(dev), int(ordinal))
                or cu.cuDeviceTotalMem_v2(ctypes.byref(total), dev)):
            return None
        return int(total.value)
    except Exception:                                         # noqa: BLE001
        return None
#: allocator -> whether the PJRT client keeps arena accounting, i.e.
#: whether ``memory_stats()['peak_bytes_in_use']`` means anything.
_XLA_PEAK_ACCOUNTING = {
    "default":    "arena",
    "bfc":        "arena",
    "cuda_async": "arena",
    "platform":   "none",
}
_XLA_PEAK_NOTE = {
    "default":    "BFC arena; peak_bytes_in_use is the exact high-water mark.",
    "bfc":        "BFC arena; peak_bytes_in_use is the exact high-water mark.",
    "cuda_async": ("cudaMallocAsync; peak_bytes_in_use measured IDENTICAL to "
                   "BFC for steady allocations (job 7882447).  Transient-heavy "
                   "kernels were not tested — treat a peak from this allocator "
                   "as unverified there, not as wrong."),
    "platform":   ("plain cudaMalloc (NOT cudaMallocAsync); the client reports "
                   "bytes_limit=0 and peak_bytes_in_use=0, so there is no "
                   "arena peak at all."),
}


@dataclass(frozen=True)
class XlaGpuMemoryEnv:
    """Resolved view of the XLA GPU-pool env, as jax will read it."""

    allocator: str
    allocator_raw: str | None
    allocator_is_valid: bool
    peak_accounting: str            # "arena" | "none" | "unknown"
    peak_is_faithful: bool
    peak_note: str
    preallocate: bool
    preallocate_raw: str | None
    preallocate_looks_like_a_typo: bool
    mem_fraction: str | None
    mem_fraction_var: str | None
    mem_fraction_deprecated: bool
    mem_fraction_conflict: bool

    def caveat(self) -> str:
        """One-line caveat for a reported GPU peak, or ``""`` when clean."""
        if not self.allocator_is_valid:
            return (f"  [XLA_PYTHON_CLIENT_ALLOCATOR={self.allocator_raw!r} is "
                    f"not one of {'/'.join(_XLA_ALLOCATORS)} — jax refuses "
                    f"this at backend init; peak reliability unknown]")
        if self.peak_accounting == "none":
            return ("  [the platform allocator reports bytes_limit=0 and "
                    "peak_bytes_in_use=0, so this figure did NOT come from "
                    "the XLA arena — it is an nvidia-smi sample of the whole "
                    "GPU.  Unset XLA_PYTHON_CLIENT_ALLOCATOR for a real peak]")
        return ""


def resolve_xla_gpu_memory_env() -> XlaGpuMemoryEnv:
    """Resolve the XLA GPU-pool env exactly as jaxlib 0.9 resolves it.

    Pure: reads ``os.environ`` and nothing else, imports no jax, and is
    safe before backend init.  The values are only MEANINGFUL on the CUDA
    backend — ``generate_pjrt_gpu_plugin_options`` is the CUDA plugin's
    option builder and no CPU code path reads any of these — so callers
    must qualify what they print by the live backend.

    IMPORTANT: this is the ENVIRONMENT's story, which is not always the
    client's.  ``os.environ`` is a false witness for allocator state — see
    :func:`classify_xla_pool`, which corroborates it against the live
    device, and use that wherever a NUMBER is being reported.
    """
    alloc_raw = os.environ.get("XLA_PYTHON_CLIENT_ALLOCATOR")
    alloc = (alloc_raw if alloc_raw is not None else "default").strip().lower()
    alloc_valid = alloc in _XLA_ALLOCATORS
    accounting = _XLA_PEAK_ACCOUNTING.get(alloc, "unknown") if alloc_valid \
        else "unknown"
    note = _XLA_PEAK_NOTE.get(alloc, "") if alloc_valid else (
        f"{alloc!r} is not an allocator jax accepts.")

    prealloc_raw = os.environ.get("XLA_PYTHON_CLIENT_PREALLOCATE")
    if prealloc_raw is None or prealloc_raw == "":
        preallocate = True                      # jax leaves the option unset
        prealloc_typo = False
    else:
        preallocate = prealloc_raw not in ("false", "False", "0")
        # A value that a human reads as "off" but jax reads as "on".
        prealloc_typo = (preallocate
                         and prealloc_raw.strip().lower() in _ENV_FALSE)

    cur = os.environ.get("XLA_CLIENT_MEM_FRACTION") or ""
    dep = os.environ.get("XLA_PYTHON_CLIENT_MEM_FRACTION") or ""
    if cur:
        frac, frac_var, frac_dep = cur, "XLA_CLIENT_MEM_FRACTION", False
    elif dep:
        frac, frac_var, frac_dep = dep, "XLA_PYTHON_CLIENT_MEM_FRACTION", True
    else:
        frac, frac_var, frac_dep = None, None, False

    return XlaGpuMemoryEnv(
        allocator=alloc,
        allocator_raw=alloc_raw,
        allocator_is_valid=alloc_valid,
        peak_accounting=accounting,
        peak_is_faithful=(accounting == "arena"),
        peak_note=note,
        preallocate=preallocate,
        preallocate_raw=prealloc_raw,
        preallocate_looks_like_a_typo=prealloc_typo,
        mem_fraction=frac,
        mem_fraction_var=frac_var,
        mem_fraction_deprecated=frac_dep,
        mem_fraction_conflict=bool(cur and dep),
    )


# ---------------------------------------------------------------------------
#  The CLIENT, not the environment
# ---------------------------------------------------------------------------
#
# ``os.environ`` IS A FALSE WITNESS FOR ALLOCATOR STATE.  Measured by the
# allocator workstream (job 7882443): ``kin_ion_io`` pre- and post-refactor
# ended with IDENTICAL ``os.environ`` but DIFFERENT clients —
# ``bytes_limit`` 11.805 GB vs 0.000 GB.  The mechanism is ordering:
# ``runtime.bootstrap()`` calls ``fallback_to_cpu_if_no_gpu_backend()``,
# which calls ``jax.devices()``, and *that* is backend init.  A
# ``setdefault`` twenty lines later sets a string and changes nothing —
# the client was already built.
#
# So anything that reports a memory NUMBER has to corroborate the
# environment against the live device rather than trust the strings.
# :func:`classify_xla_pool` is the pure half of that check (it takes a
# ``memory_stats()`` dict, so it stays importable without jax and is
# testable on a login node); the callers supply the live stats.

@dataclass(frozen=True)
class XlaPoolReading:
    """What the LIVE client says, cross-checked against the environment."""

    accounting_present: bool
    bytes_limit: int
    peak_bytes_in_use: int
    #: "arena"      — the printed peak came from XLA's own accounting
    #: "nvidia-smi" — it came from the whole-GPU fallback sample
    #: "none"       — there is no peak to print
    peak_source: str
    env_agrees: bool
    disagreement: str


def classify_xla_pool(stats, *, backend: str = "gpu",
                      env: XlaGpuMemoryEnv | None = None) -> XlaPoolReading:
    """Corroborate the XLA memory environment against the live client.

    Parameters
    ----------
    stats
        ``jax.local_devices()[0].memory_stats()`` — ``None``/``{}`` when
        the backend keeps no arena accounting.
    backend
        ``jax.default_backend()``.  On a non-GPU backend the ABSENCE of
        accounting is normal and is NOT reported as a disagreement:
        crying wolf on every CPU run is how a warning stops being read.
    """
    xm = env if env is not None else resolve_xla_gpu_memory_env()
    st = dict(stats or {})
    try:
        limit = int(st.get("bytes_limit", 0) or 0)
        peak = int(st.get("peak_bytes_in_use", 0) or 0)
    except (TypeError, ValueError):
        limit, peak = 0, 0
    present = bool(limit > 0 or peak > 0)

    is_gpu = str(backend).strip().lower() in ("gpu", "cuda", "rocm")
    if not is_gpu:
        return XlaPoolReading(
            accounting_present=present, bytes_limit=limit,
            peak_bytes_in_use=peak,
            peak_source="arena" if present else "none",
            env_agrees=True, disagreement="")

    expects_arena = (xm.peak_accounting == "arena")
    disagreement = ""
    if expects_arena and not present:
        disagreement = (
            f"XLA_PYTHON_CLIENT_ALLOCATOR resolves to {xm.allocator!r}, which "
            f"keeps arena accounting, but the live client reports "
            f"bytes_limit={limit} peak_bytes_in_use={peak}.  The allocator is "
            f"fixed at backend init — bootstrap()'s "
            f"fallback_to_cpu_if_no_gpu_backend() calls jax.devices() — so a "
            f"variable set after that point changes the string and not the "
            f"client (measured, job 7882443).  Every memory number in this "
            f"run is therefore NOT from the allocator the environment names.")
    elif (not expects_arena) and present and xm.allocator_is_valid:
        disagreement = (
            f"XLA_PYTHON_CLIENT_ALLOCATOR resolves to {xm.allocator!r}, which "
            f"reports no arena, but the live client reports "
            f"bytes_limit={limit}.  The client was built before this variable "
            f"took effect (backend init ordering); trust the client.")

    if present:
        source = "arena"
    elif xm.peak_accounting == "none":
        source = "nvidia-smi"       # what _track_peak falls back to
    else:
        source = "none"
    return XlaPoolReading(
        accounting_present=present, bytes_limit=limit, peak_bytes_in_use=peak,
        peak_source=source, env_agrees=(disagreement == ""),
        disagreement=disagreement)



# ---------------------------------------------------------------------------
#  The pool high-water mark: a per-stage device peak that resets
# ---------------------------------------------------------------------------
#
# ``peak_bytes_in_use`` never resets, so a stage below an earlier high-water
# mark is invisible (lane MEM, claim 2804).  The CUDA pool behind XLA's
# ``cuda_async`` allocator keeps ``CU_MEMPOOL_ATTR_USED_MEM_HIGH``, which a
# set of 0 resets to the bytes in use now.  Its handle is found once, from
# one live buffer (``CU_POINTER_ATTRIBUTE_MEMPOOL_HANDLE``).  A read is then
# two driver calls and needs no context and no synchronization.  It counts
# what XLA allocates from the pool (executables' temporaries, the compile's
# autotuner buffers) and not NCCL or library workspaces outside it.

_CU_POINTER_ATTRIBUTE_MEMPOOL_HANDLE = 17
_CU_MEMPOOL_ATTR_USED_MEM_HIGH = 8
_POOL: dict = {"lib": None, "handle": None, "source": None}


def _find_pool():
    """The pool handle of this process's device, or None (not a GPU, the backend not up
    yet, or an allocator without a pool).  Never initializes the backend itself."""
    import sys
    bridge = sys.modules.get("jax._src.xla_bridge")
    if not (bridge and getattr(bridge, "_backends", None)):
        return None
    import ctypes
    import jax
    if jax.local_devices()[0].platform != "gpu":
        _POOL["source"] = "none"
        return None
    pointer = None
    for array in jax.live_arrays():          # one device buffer; nothing is allocated
        try:
            if (array.nbytes and not array.is_deleted()
                    and getattr(array.sharding, "memory_kind", None) in (None, "device")):
                pointer = array.addressable_shards[0].data.unsafe_buffer_pointer()
                break
        except Exception:                                      # noqa: BLE001
            continue
    if pointer is None:
        return None                          # no live buffer yet: ask again next time
    lib = ctypes.CDLL("libcuda.so.1")
    handle = ctypes.c_void_p()
    rc = lib.cuPointerGetAttribute(ctypes.byref(handle),
                                   _CU_POINTER_ATTRIBUTE_MEMPOOL_HANDLE,
                                   ctypes.c_uint64(pointer))
    if rc == 0 and handle.value:
        _POOL.update(lib=lib, handle=handle, source="pool")
        return handle
    _POOL["source"] = "peak_bytes_in_use"
    import warnings
    warnings.warn("device peaks per stage fall back to peak_bytes_in_use: the allocator "
                  "keeps no CUDA pool, and that figure never resets, so a stage below an "
                  "earlier high-water mark reads as the earlier mark.", RuntimeWarning)
    return None


def pool_high_water(*, reset: bool = False) -> int | None:
    """Bytes in use on this process's device at their highest since the last reset.

    ``reset`` then sets the mark back to the bytes in use now.  On an allocator
    without a CUDA pool it returns ``peak_bytes_in_use``, which never resets
    (a warning says so once).  None before the backend is up and on CPU.
    """
    try:
        if _POOL["source"] is None:
            _find_pool()
        if _POOL["handle"] is not None:
            import ctypes
            value = ctypes.c_uint64(0)
            lib, handle = _POOL["lib"], _POOL["handle"]
            if lib.cuMemPoolGetAttribute(handle, _CU_MEMPOOL_ATTR_USED_MEM_HIGH,
                                         ctypes.byref(value)):
                return None
            high = int(value.value)
            if reset:
                value.value = 0
                lib.cuMemPoolSetAttribute(handle, _CU_MEMPOOL_ATTR_USED_MEM_HIGH,
                                          ctypes.byref(value))
            return high
        if _POOL["source"] == "peak_bytes_in_use":
            import jax
            stats = jax.local_devices()[0].memory_stats() or {}
            return int(stats.get("peak_bytes_in_use", 0)) or None
    except Exception:                                          # noqa: BLE001
        return None      # an instrument never takes down the run it measures
    return None


def pool_high_water_source() -> str | None:
    """What :func:`pool_high_water` reads: "pool", "peak_bytes_in_use", "none", or None
    before the first reading."""
    return _POOL["source"]
