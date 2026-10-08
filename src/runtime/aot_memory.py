"""True per-rank peak HBM for a compiled JAX kernel — cuFFT scratch included.

``compiled.memory_analysis()`` reports ONLY what XLA's buffer assignment
placed (``temp + argument + output - alias``).  A cuFFT plan's *workspace*
is not in buffer assignment at all: jaxlib's ``FftThunk`` takes it from a
runtime scratch allocator at execution time.  A peak read from
``memory_analysis()`` alone is therefore a systematic UNDER-estimate for any
kernel containing an FFT, growing with the FFT box.  Measured size of that
gap: docs/architecture/memory-model.md §"cuFFT plan scratch".

Three steps close it: ``memory_analysis()`` -> ``compiled_peak``; a regex
over ``compiled.as_text()`` -> one :class:`FftSpec` per fft op;
``cufftMakePlanMany`` (autoAllocation off) on the *exact* libcufft jaxlib
``dlopen``'d, found via ``/proc/self/maps`` -> ``cufft_scratch``.  Using
jaxlib's own libcufft is what makes the plan we size the plan the runtime
will build — no version drift between query and execution.

Public API: :func:`aot_kernel_peak_bytes` ``(compiled, platform=None) ->
AotPeakBreakdown``; compare ``.total`` against the per-rank budget.  The
callers are the stage memory planners that compile their own kernel
(``common.wfn_transforms``, ``isdf.core``, ``gw.mpa.sigma``,
``gw.mpa.sector_sigma``).

:func:`check_chunk` is the one compiled check of an analytic chunk: a
planner's closed form chooses the chunk, the chosen executable is compiled
anyway, and its figure corrects the per-unit slope, re-solves the chunk and
recompiles once only when it exceeds the room; :func:`agreed_chunk` makes
the chunk every rank runs the same (the response direct stream, the ζ μ batch; the Σ τ window reads
:func:`compiled_new_bytes`).

Works on CPU and GPU.  On a non-CUDA platform there are no cuFFT plans, so
the scratch term is an exact 0 — NOT a demotion.  That has to be decided
from the platform, not the HLO: measured on jax 0.9.1 (job 7882062),
XLA:CPU keeps the ``fft`` op in ``as_text()`` exactly like XLA:GPU, so
"the HLO has an fft op" does not mean "cuFFT is involved".

No silent demotions.  An HLO regex miss raises :class:`HloFftParseError`
rather than reading as "no FFTs, no scratch".  An absent libcufft or a
refused plan *on a CUDA backend* gives ``cufft_scratch = 0`` **with**
``cufft_measured = False`` **and** an announcement from the process it
happened on; a caller that must not under-predict checks the flag instead
of trusting ``total``.
"""

from __future__ import annotations

import ctypes
import functools
import os
import re
from dataclasses import dataclass

from runtime import _resolve_proc_id


_announced: set[str] = set()


def announce_once(key: str, message: str) -> None:
    """Print ``message`` once per process, tagged with the process index.

    Standing doctrine: a demotion may happen, but it must announce itself
    from the rank it happens on.  Every path in the memory model that returns
    a weaker number than it advertises comes through here, so the demotion
    lands in that rank's log instead of nowhere.  ``key`` dedupes per process
    so a per-shape loop cannot spam the log.
    """
    if key in _announced:
        return
    _announced.add(key)
    print(f"*** [memory-model][proc {_resolve_proc_id()}] {message} ***",
          flush=True)


@dataclass(frozen=True)
class FftSpec:
    """One cuFFT plan, as jaxlib's FFT thunk would build it: ``rank``
    transform dims of extents ``transform_shape`` (row-major — XLA's
    ``fft_length``, cuFFT's ``n[]``), ``batch`` = product of the leading
    (non-transform) operand dims, ``dtype`` in ``c128/c64/f64/f32``,
    ``fft_type`` in ``FFT/IFFT/RFFT/IRFFT``."""
    rank: int
    transform_shape: tuple[int, ...]
    batch: int
    dtype: str
    fft_type: str


@dataclass(frozen=True)
class AotPeakBreakdown:
    """Per-rank peak prediction, split by source.

    ``compiled_peak``  ``temp + argument + output - alias`` from
                       ``memory_analysis()`` — XLA-visible buffers only.
    ``cufft_scratch``  max plan workspace over the distinct FFT ops (only
                       one plan's scratch is live at a time inside a kernel).
    ``total``          ``compiled_peak + cufft_scratch`` — compare to budget.
    ``cufft_measured`` ``False`` means the cuFFT query was unavailable, so
                       ``cufft_scratch = 0`` is a KNOWN-LOW placeholder.
    ``fft_specs``      the plans parsed out of the HLO.
    ``resident_increment`` extra bytes above arguments that a caller has
                       already counted as resident: ``temp + output - alias``
                       plus cuFFT scratch.
    """
    compiled_peak: int
    cufft_scratch: int
    total: int
    cufft_measured: bool
    fft_specs: tuple[FftSpec, ...]
    resident_increment: int = 0


class HloFftParseError(RuntimeError):
    """The HLO FFT-op regex failed on a form it should have matched.  Loud by
    design: degrading to ``compiled_peak`` alone silently reintroduces the
    under-prediction this module exists to remove."""


class CufftQueryError(RuntimeError):
    """``cufftCreate``/``cufftMakePlanMany`` failed, or libcufft is absent."""


# ---------------------------------------------------------------------------
# HLO FFT-op parser
# ---------------------------------------------------------------------------

# JAX/XLA prints the lowered fft op in two flavours; both put the *output*
# dtype and shape on the LHS of ``=``, which is all a C2C plan needs:
#
#   (A) typed operands (verbose):
#       %fft.1 = c128[10,75,75,200]{3,2,1,0} fft(c128[10,75,75,200]{3,2,1,0}
#           %arg.0), fft_type=FFT, fft_length={75,75,200}
#   (B) untyped operands (current ``compiled.as_text()``):
#       ROOT %fft.0 = c64[10,60,60,80]{3,2,1,0} fft(%x.1),
#           fft_type=FFT, fft_length={60,60,80}, metadata={...}
#
# R2C/C2R carry a half-spectrum output shape; the cuFFT type code accounts
# for it, so we do not enforce shape equality for those.
_FFT_OP_RE = re.compile(
    r"""
    %\S+ \s*=\s*                            # SSA result name
    (?P<dtype>[a-z]+\d+)                    # 'c128' / 'c64' / 'f64' / 'f32'
    \s* \[ (?P<shape>[\d,\s]+) \]           # output shape
    (?:\{[\d,\s]*\})?                       # optional layout
    \s* fft\s*\( [^)]* \)                   # the 'fft(' call
    \s*,\s* fft_type \s*=\s* (?P<fft_type>FFT|IFFT|RFFT|IRFFT)
    \s*,\s* fft_length \s*=\s* \{ (?P<fft_length>[\d,\s]+) \}
    """,
    re.VERBOSE,
)


def parse_fft_specs_from_hlo(hlo_text: str) -> list[FftSpec]:
    """One :class:`FftSpec` per fft op in ``hlo_text``.

    An empty list is a valid result — the kernel has no FFTs.  But if the
    text contains ``" fft("`` and the regex matches nothing we raise
    :class:`HloFftParseError`: format drift must not read as "no FFTs, no
    scratch".

    MEASURED on jax 0.9.1 (job 7882062): BOTH XLA:GPU and XLA:CPU keep the
    ``fft`` op in ``compiled.as_text()``, so specs parse on either platform.
    A parsed spec therefore does NOT imply cuFFT — see
    :func:`aot_kernel_peak_bytes`, which decides that from the platform.
    """
    specs: list[FftSpec] = []
    for m in _FFT_OP_RE.finditer(hlo_text):
        op_shape = tuple(int(x) for x in m.group("shape").split(",") if x.strip())
        fft_length = tuple(
            int(x) for x in m.group("fft_length").split(",") if x.strip())
        rank = len(fft_length)
        fft_type = m.group("fft_type")
        if rank not in (1, 2, 3):
            raise HloFftParseError(
                f"fft_length={fft_length} has rank {rank}; cuFFT does 1/2/3.")
        if len(op_shape) < rank:
            raise HloFftParseError(
                f"Output shape {op_shape} has fewer dims than fft rank {rank} "
                f"— HLO format may have shifted.")
        if fft_type in ("FFT", "IFFT") and op_shape[-rank:] != fft_length:
            raise HloFftParseError(
                f"C2C output shape {op_shape} trailing dims != "
                f"fft_length={fft_length}; HLO format may have shifted.")
        batch = 1
        for d in op_shape[:-rank]:
            batch *= d
        specs.append(FftSpec(rank=rank, transform_shape=fft_length,
                             batch=batch, dtype=m.group("dtype"),
                             fft_type=fft_type))

    if " fft(" in hlo_text and not specs:
        raise HloFftParseError(
            "HLO text contains ' fft(' but the FFT-op regex matched zero ops. "
            "XLA may have changed its lowered HLO format.  Update _FFT_OP_RE "
            "in src/runtime/aot_memory.py.")
    return specs


# ---------------------------------------------------------------------------
# cuFFT plan-workspace query
# ---------------------------------------------------------------------------

# cufftType codes and the success sentinel.  Source: cuda/include/cufft.h.
_CUFFT_R2C, _CUFFT_C2R, _CUFFT_C2C = 0x2A, 0x2C, 0x29   # single precision
_CUFFT_D2Z, _CUFFT_Z2D, _CUFFT_Z2Z = 0x6A, 0x6C, 0x69   # double precision
_CUFFT_SUCCESS = 0

# (operand dtype, XLA op kind) -> cuFFT type code.  FFT and IFFT share a
# plan: direction is a runtime argument, not a plan property.
_CUFFT_TYPE = {
    ("c128", "FFT"): _CUFFT_Z2Z, ("c128", "IFFT"): _CUFFT_Z2Z,
    ("c128", "RFFT"): _CUFFT_D2Z, ("c128", "IRFFT"): _CUFFT_Z2D,
    ("c64", "FFT"): _CUFFT_C2C, ("c64", "IFFT"): _CUFFT_C2C,
    ("c64", "RFFT"): _CUFFT_R2C, ("c64", "IRFFT"): _CUFFT_C2R,
    ("f64", "RFFT"): _CUFFT_D2Z, ("f32", "RFFT"): _CUFFT_R2C,
}


def _cufft_type_for(dtype: str, fft_type: str) -> int:
    try:
        return _CUFFT_TYPE[(dtype, fft_type)]
    except KeyError:
        raise CufftQueryError(
            f"Unsupported (dtype={dtype}, fft_type={fft_type}) combination."
        ) from None


@functools.lru_cache(maxsize=1)
def _jax_cufft_handle() -> ctypes.CDLL:
    """``ctypes``-load the ``libcufft.so`` jaxlib itself already loaded.

    Forces CUDA platform init first so the library is actually mapped, then
    reads ``/proc/self/maps`` for it.  Using that exact path (rather than a
    system libcufft) is what makes the queried workspace equal the workspace
    the runtime FFT thunk will allocate.

    Raises :class:`CufftQueryError` when there is no CUDA backend or the
    library is not in the process map (e.g. a statically linked jaxlib).
    """
    import jax
    try:
        jax.devices("gpu")          # forces CUDA platform init; no-op if done
    except Exception as e:
        raise CufftQueryError(
            f"no CUDA backend; cannot query cuFFT scratch: {e}") from e

    # Lines look like:
    #   7f12...-7f12... r-xp 00000000 fd:01 12345  /path/to/libcufft.so.12
    cufft_path = None
    try:
        with open("/proc/self/maps", "r") as fh:
            for line in fh:
                parts = line.rstrip().split(None, 5)
                if len(parts) == 6 and parts[5].startswith("/") \
                        and os.path.basename(parts[5]).startswith("libcufft.so"):
                    cufft_path = parts[5]
                    break
    except OSError as e:
        raise CufftQueryError(f"could not read /proc/self/maps: {e}") from e
    if cufft_path is None:
        raise CufftQueryError(
            "libcufft.so is not in /proc/self/maps after CUDA init — a "
            "CPU-only or statically linked jaxlib build?")

    try:
        lib = ctypes.CDLL(cufft_path, mode=ctypes.RTLD_GLOBAL)
    except OSError as e:
        raise CufftQueryError(f"failed to ctypes-load {cufft_path}: {e}") from e

    # ABI signatures, cuda/include/cufft.h.  inembed/onembed are passed NULL
    # (packed contiguous), matching jaxlib's FFT thunk for a batched C2C box.
    lib.cufftCreate.restype = ctypes.c_int
    lib.cufftCreate.argtypes = [ctypes.POINTER(ctypes.c_int)]
    lib.cufftDestroy.restype = ctypes.c_int
    lib.cufftDestroy.argtypes = [ctypes.c_int]
    lib.cufftSetAutoAllocation.restype = ctypes.c_int
    lib.cufftSetAutoAllocation.argtypes = [ctypes.c_int, ctypes.c_int]
    lib.cufftMakePlanMany.restype = ctypes.c_int
    lib.cufftMakePlanMany.argtypes = [
        ctypes.c_int, ctypes.c_int, ctypes.POINTER(ctypes.c_int),   # plan/rank/n
        ctypes.POINTER(ctypes.c_int), ctypes.c_int, ctypes.c_int,   # in embed/stride/dist
        ctypes.POINTER(ctypes.c_int), ctypes.c_int, ctypes.c_int,   # out embed/stride/dist
        ctypes.c_int, ctypes.c_int,                                 # type, batch
        ctypes.POINTER(ctypes.c_size_t),                            # workSize
    ]
    return lib


@functools.lru_cache(maxsize=512)
def _query_one_plan_workspace_bytes(spec: FftSpec) -> int:
    """Plan-workspace bytes for ``spec``, via ``cufftMakePlanMany``.

    ``cufftSetAutoAllocation(plan, 0)`` means no device memory beyond the
    plan descriptor is allocated — the query is cheap and side-effect free.
    Queried at the *production* batch: cuFFT picks its algorithm from the
    plan parameters, so extrapolating from a small batch would be a
    different algorithm's workspace.
    """
    lib = _jax_cufft_handle()
    if spec.batch >= (1 << 31):
        raise CufftQueryError(
            f"batch={spec.batch} exceeds int32; needs cufftMakePlanMany64.")
    ctype_code = _cufft_type_for(spec.dtype, spec.fft_type)

    # idist/odist are per-batch element counts: product(n) for C2C, with the
    # half-spectrum (n[-1]//2 + 1) substitution on the complex side of
    # R2C/C2R — the convention jaxlib's fft_thunk passes.
    dist = 1
    for ni in spec.transform_shape:
        dist *= int(ni)
    half = (dist // int(spec.transform_shape[-1])) * (
        int(spec.transform_shape[-1]) // 2 + 1)
    in_dist = half if spec.fft_type == "IRFFT" else dist
    out_dist = half if spec.fft_type == "RFFT" else dist

    plan = ctypes.c_int(0)
    rc = lib.cufftCreate(ctypes.byref(plan))
    if rc != _CUFFT_SUCCESS:
        raise CufftQueryError(f"cufftCreate failed: rc={rc}")
    try:
        rc = lib.cufftSetAutoAllocation(plan, 0)
        if rc != _CUFFT_SUCCESS:
            raise CufftQueryError(f"cufftSetAutoAllocation failed: rc={rc}")
        work_size = ctypes.c_size_t(0)
        rc = lib.cufftMakePlanMany(
            plan, spec.rank, (ctypes.c_int * spec.rank)(*spec.transform_shape),
            None, 1, in_dist, None, 1, out_dist,
            ctype_code, spec.batch, ctypes.byref(work_size))
        if rc != _CUFFT_SUCCESS:
            raise CufftQueryError(
                f"cufftMakePlanMany failed: rc={rc} for spec={spec} "
                f"(idist={in_dist}, odist={out_dist}, type=0x{ctype_code:x})")
        return int(work_size.value)
    finally:
        lib.cufftDestroy(plan)


def query_cufft_workspace_bytes(specs) -> int:
    """**Max** plan workspace over the distinct ``FftSpec``s in ``specs``.

    Max, not sum: XLA schedules FFT thunks sequentially, so within one
    kernel only one plan's scratch is live at a time.  "Distinct" is spec
    equality — same transform shape + batch + dtype + op kind is the same
    plan.  Returns 0 for an empty ``specs``.
    """
    distinct = set(specs)
    if not distinct:
        return 0
    return max(_query_one_plan_workspace_bytes(s) for s in distinct)


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def _is_cuda_platform(platform: str | None) -> bool:
    """Is the executable's platform one where cuFFT plans exist?

    ``platform`` is the caller's declaration (``mesh.devices[0].platform``);
    a GPU is CUDA when this process's device vendor is
    (:func:`lxkit.device_vendor`).
    """
    if platform is not None and platform not in ("gpu", "cuda"):
        return False
    try:
        import jax
        from lxkit import device_vendor
        return device_vendor(jax.local_devices()[0]) == "cuda"
    except Exception:
        return False


def buffer_peak_bytes(m) -> int:
    """``temp + argument + output - alias`` of one ``CompiledMemoryStats``."""
    return (int(m.temp_size_in_bytes) + int(m.argument_size_in_bytes)
            + int(m.output_size_in_bytes) - int(m.alias_size_in_bytes))


class ModuleDoesNotFit(RuntimeError):
    """GATE xla_rematerialization: a compiled module is larger than the device."""


@functools.lru_cache(maxsize=None)
def _device_total(ordinal: int) -> int | None:
    from runtime.xla_memory import cuda_device_total_bytes
    return cuda_device_total_bytes(ordinal)


def refuse_over_device(executable, module_name: str, platform: str) -> None:
    """Refuse a compiled module whose buffers exceed the device.

    Called once per compiled module, when jax builds its executor for the
    first execution (``common.jax_compile_cache._install_device_fit_gate``).
    XLA's rematerialization pass is off
    (``runtime.disable_xla_rematerialization``), so nothing at compile time
    shrinks a module that does not fit; it would fail at its first
    allocation.  This refuses it at once instead.  The limit is the card's
    total memory (``cuDeviceTotalMem``), a hard device limit: the buffers
    alone (``temp + arguments + outputs - aliased``, no cuFFT scratch, no
    NCCL or context) cannot fit above it.  It is not the memory budget;
    ``memory_per_device_gb`` still only warns.  CUDA only.
    """
    if (platform not in ("cuda", "gpu")
            or not hasattr(executable, "get_compiled_memory_stats")):
        return
    stats = executable.get_compiled_memory_stats()
    peak = buffer_peak_bytes(stats)
    try:
        ordinal = int(executable.local_devices()[0].local_hardware_id or 0)
    except Exception:                                      # noqa: BLE001
        ordinal = 0
    total = _device_total(ordinal)
    if total is None or peak <= total:
        return
    raise ModuleDoesNotFit(
        f"GATE xla_rematerialization: REFUSED module {module_name!r}: its "
        f"compiled buffers need {peak / 1e9:.2f} GB per device (temp "
        f"{int(stats.temp_size_in_bytes) / 1e9:.2f} + arguments "
        f"{int(stats.argument_size_in_bytes) / 1e9:.2f} + outputs "
        f"{int(stats.output_size_in_bytes) / 1e9:.2f} - aliased "
        f"{int(stats.alias_size_in_bytes) / 1e9:.2f}), and the device has "
        f"{total / 1e9:.2f} GB.  XLA's rematerialization pass is off in "
        f"LORRAX, so this module cannot fit; it was stopped before it ran.  "
        f"Reduce its peak: more ranks (most peaks fall as 1/P), fewer "
        f"centroids or bands, or a smaller memory_per_device_gb where the "
        f"stage's planner sizes from it.  See "
        f"docs/architecture/memory-model.md#module-does-not-fit.")


def aot_kernel_peak_bytes(compiled, *, platform: str | None = None
                          ) -> AotPeakBreakdown:
    """Per-rank peak HBM for ``compiled`` (a ``jax.stages.Compiled``),
    XLA-visible buffers **plus** the cuFFT plan workspace XLA cannot see.

    ``platform`` declares the executable's backend (``"gpu"``/``"cuda"`` vs
    anything else); leave it ``None`` to infer from this process's devices.
    It decides whether a zero scratch term is a *fact* or a *demotion* —
    and that distinction is not derivable from the HLO, because (measured on
    jax 0.9.1, job 7882062) XLA:CPU keeps the ``fft`` op in ``as_text()``
    exactly like XLA:GPU does:

    * non-CUDA -> ``cufft_scratch = 0``, ``cufft_measured = True``.  There
      are no cuFFT plans on this platform, so 0 is exact.  (XLA:CPU's own
      Ducc FFT scratch is a different, much smaller quantity that this
      module does not claim to model.)
    * CUDA, query succeeds -> the measured workspace, ``measured = True``.
    * CUDA, query fails -> 0 with ``measured = False``, announced once.
    """
    m = compiled.memory_analysis()
    compiled_peak = buffer_peak_bytes(m)
    # Some planners query an executable whose arguments are already part of
    # their measured resident floor (a shard-local slice is the motivating
    # case).  ``resident_increment`` is the exact extra buffer-assignment
    # footprint above those live arguments; ``total`` remains the standalone
    # kernel peak used by callers whose arguments are not otherwise priced.
    compiled_increment = max(
        0,
        int(m.temp_size_in_bytes)
        + int(m.output_size_in_bytes)
        - int(m.alias_size_in_bytes),
    )

    fft_specs = tuple(parse_fft_specs_from_hlo(compiled.as_text()))

    cufft_scratch, measured = 0, True
    if fft_specs and _is_cuda_platform(platform):
        try:
            cufft_scratch = query_cufft_workspace_bytes(fft_specs)
        except CufftQueryError as exc:
            measured = False
            announce_once(
                f"cufft-query-unavailable:{type(exc).__name__}",
                f"cuFFT plan-workspace query UNAVAILABLE on a CUDA backend "
                f"({exc}).  Peak predictions for the {len(fft_specs)} FFT "
                f"op(s) in this kernel omit the plan workspace and are "
                f"therefore LOW BOUNDS — at the CrI3 V_q box that term was "
                f">13.7 GB/rank")

    return AotPeakBreakdown(
        compiled_peak=compiled_peak,
        cufft_scratch=cufft_scratch,
        total=compiled_peak + cufft_scratch,
        cufft_measured=measured,
        fft_specs=fft_specs,
        resident_increment=compiled_increment + cufft_scratch,
    )


# ---------------------------------------------------------------------------
# One compiled check of an analytic chunk -- every stage planner calls this
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ChunkCheck:
    """What :func:`check_chunk` decided.

    ``chunk``/``compiled`` are what runs.  ``analytic`` is the planner's price
    ``fixed + chunk0·per_unit`` of the chunk it chose, ``compiled_bytes`` the
    executable's figure for that chunk, ``price`` the figure of the chunk that
    runs (compiled when it was higher), ``per_unit`` the slope that chose it,
    ``seconds`` the wall of reading the figures (memory analysis, HLO parse,
    cuFFT plan query), compiles excluded.
    """
    chunk: int
    compiled: object
    analytic: int
    compiled_bytes: int
    price: int
    per_unit: float
    recompiled: bool
    seconds: float = 0.0


def compiled_new_bytes(compiled, *, extra: int = 0, platform: str | None = None) -> int:
    """New bytes per rank of ``compiled`` beside its live arguments:
    ``temp + outputs - alias`` plus cuFFT plan scratch, plus ``extra`` (what the
    stage adds that buffer assignment cannot see: donated carries allocated by
    the caller, a native handler's run-time scratch)."""
    return int(aot_kernel_peak_bytes(compiled, platform=platform).resident_increment) + int(extra)


def agreed_chunk(chunk: int) -> int:
    """The smallest of every rank's value (one small all-gather).

    A compiled figure is read on each rank and can differ between them (a
    cuFFT scratch query that came back 0 on one), so :func:`check_chunk` and
    :func:`step_up` compare the largest figure (``-agreed_chunk(-figure)``)
    and every rank compiles and runs the same chunk (INVARIANTS 21).
    """
    import numpy as np
    from common.collectives import all_gather_processes
    return int(np.min(np.asarray(all_gather_processes(np.asarray(int(chunk), dtype=np.int64)))))


def step_up(value: int, top: int, *, build, compiled, figure, room: float, stage: str = "",
            snap=lambda v: v):
    """Raise ``value`` (to at most ``top``) while the compiled figure is over
    ``room`` and each step lowers it; past that something the value does not
    touch binds.  Each step goes to the value the figure implies, bytes =
    a + b/value (``a`` from the last two figures, 0 at the first; doubling
    when ``a`` is over the room), rounded up by ``snap`` to the caller's next
    distinct value.  ``build(v)`` compiles at ``v``, ``figure(compiled)`` reads
    its bytes.  Every rank decides on the largest figure, so all ranks step in
    lockstep (INVARIANTS 21).  Returns ``(value, compiled)`` that run."""
    agreed_max = lambda ex: -agreed_chunk(-int(figure(ex)))
    got, prev = agreed_max(compiled), None
    while got > room and value < top:
        a = 0.0 if prev is None else (got * value - prev[1] * prev[0]) / (value - prev[0])
        want = -(-value * (got - a) // (room - a)) if room > a else 2 * int(value)
        # Clamp before snapping: the caller's snap is defined only up to ``top``
        # (route G's divides by zero past it).
        nxt_v = min(snap(min(max(int(want), int(value) + 1), int(top))), int(top))
        nxt_c = build(nxt_v)
        nxt = agreed_max(nxt_c)
        if nxt >= got:
            break
        prev, (value, compiled, got) = (value, got), (nxt_v, nxt_c, nxt)
        announce_once(f"step-up:{stage}:{value}", f"{stage}: compiled {got / 1e9:.2f} GB at {value}")
    return value, compiled


def check_chunk(chunk: int, *, build, fixed: float, per_unit: float, room: float,
                extra=lambda c, compiled: 0, minimum: int = 1, compiled=None, stage: str = "",
                platform: str | None = None) -> ChunkCheck:
    """Check the chunk an analytic model chose against its compiled executable.

    The planner prices its stage as ``fixed + chunk·per_unit`` new bytes per
    rank beside what is live and chose ``chunk`` for ``room``.  ``build(c)``
    compiles the executable of chunk ``c`` (``compiled`` passes one already
    built for ``chunk``).  Its figure is :func:`compiled_new_bytes` plus
    ``extra(c, compiled)``: what the stage holds beside it (a donated carry
    the caller allocates, a native handler's run-time scratch, a lookahead
    copy of its output).  Within the room the chunk runs as planned: nothing
    is recompiled and results are unchanged.  The analytic price never admits
    a chunk; only the compiled figure against the room does.  Over the room,
    the slope is corrected from this one point,
    ``per_unit = max(per_unit, (compiled - fixed) / chunk)``, the chunk solved
    directly, ``floor((room - fixed) / per_unit)``, and compiled once more.
    No bisection: a chunk already at ``minimum``, or a second figure still
    over the room, prints one warning line
    (``common.gpu_utils.warn_over_budget``) and the stage runs at that chunk.
    Every figure is the largest any rank read (:func:`agreed_chunk`) before
    it is compared, so every rank recompiles, or not, in lockstep (INVARIANTS 21).
    """
    import time
    chunk = int(chunk)
    if compiled is None:
        compiled = build(chunk)
    analytic = int(fixed + chunk * per_unit)
    figure = lambda c, ex: -agreed_chunk(-compiled_new_bytes(ex, extra=int(extra(c, ex)),
                                                             platform=platform))
    started = time.perf_counter()
    got = figure(chunk, compiled)
    seconds = time.perf_counter() - started
    from common.gpu_utils import warn_over_budget
    if got <= room or chunk <= minimum:
        if got > room:
            warn_over_budget(f"{stage or 'compiled chunk'} (chunk {chunk}, compiled)", got, room)
        return ChunkCheck(chunk, compiled, analytic, got, max(analytic, got),
                          float(per_unit), False, seconds)
    slope = max(float(per_unit), (got - float(fixed)) / chunk)
    solved = max(int(minimum), min(chunk - 1, int((float(room) - float(fixed)) // slope)))
    announce_once(f"check-chunk:{stage}:{chunk}:{solved}",
                  f"{stage}: compiled {got / 1e9:.2f} GB (analytic "
                  f"{analytic / 1e9:.2f} GB) over the room {room / 1e9:.2f} GB at chunk "
                  f"{chunk}; slope {per_unit / 1e6:.1f} -> {slope / 1e6:.1f} MB per unit, "
                  f"chunk {chunk} -> {solved}, recompiled once")
    compiled = build(solved)
    started = time.perf_counter()
    again = figure(solved, compiled)
    seconds += time.perf_counter() - started
    if again > room:
        warn_over_budget(f"{stage or 'compiled chunk'} (chunk {solved}, compiled)", again, room)
    return ChunkCheck(solved, compiled, analytic, got, again, slope, True, seconds)
