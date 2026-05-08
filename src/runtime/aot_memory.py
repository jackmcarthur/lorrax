"""AOT GPU memory prediction for jax kernels with cuFFT operations.

XLA's ``compiled.memory_analysis()`` reports static + temp buffers visible
to its buffer plan, but cuFFT plan workspace is allocated lazily by
jaxlib's FFT thunk at first kernel invocation — outside XLA's accounting.
This module closes that gap by

1. taking a ``jax.stages.Compiled`` and reading its ``memory_analysis()``
   for the XLA-visible peak;
2. walking the lowered HLO for ``fft(...)`` ops and extracting plan
   parameters (rank, transform shape, batch, dtype);
3. calling ``cufftGetSize`` on the *exact* libcufft.so jaxlib has
   ``dlopen``'d (located via ``/proc/self/maps``).  Same binary jaxlib
   uses → no version drift between query-time and kernel-execution.

Public API
----------
``aot_kernel_peak_bytes(compiled)`` returns an :class:`AotPeakBreakdown`
with ``compiled_peak``, ``cufft_scratch`` (max over distinct plans), and
``total = compiled_peak + cufft_scratch``.  Caller (V_q chooser, chi0
chooser, σ chooser, bispinor pickers) compares ``total`` to its budget.

Reusable across any GW kernel that contains ``jnp.fft.fftn`` or similar.

Design choices
--------------
* HLO parser fails **loudly** on regex miss — we want to notice when XLA
  changes its FFT op format, not silently degrade to wrong predictions.
* ``cufftGetSize`` is queried at the production batch directly (not
  extrapolated from small Q); cuFFT picks its algorithm based on plan
  parameters, so this is the same scratch the runtime will allocate.
* The query path uses the libcufft ``dlopen``'d by jaxlib (resolved via
  ``/proc/self/maps`` after ``jax.devices()``), not a system libcufft —
  guarantees query and execution use the same binary.
"""

from __future__ import annotations

import ctypes
import functools
import os
import re
from dataclasses import dataclass

import jax


# ---------------------------------------------------------------------------
# Public types
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class FftSpec:
    """Identifies a cuFFT plan as jaxlib's FFT thunk would build it.

    Attributes
    ----------
    rank
        1, 2, or 3 — the number of transform dimensions.
    transform_shape
        Length-``rank`` tuple of transform extents in row-major order
        (matches XLA's ``fft_length`` and cuFFT's ``n[]``).
    batch
        Product of the leading (non-transform) operand dims.
    dtype
        One of ``'c128'``, ``'c64'``, ``'f64'``, ``'f32'``.
    fft_type
        ``'FFT'`` (forward complex), ``'IFFT'`` (inverse complex),
        ``'RFFT'`` (real-to-complex), ``'IRFFT'`` (complex-to-real).
    """
    rank: int
    transform_shape: tuple[int, ...]
    batch: int
    dtype: str
    fft_type: str


@dataclass(frozen=True)
class AotPeakBreakdown:
    """Per-rank GPU peak prediction split by source.

    Attributes
    ----------
    compiled_peak
        Bytes from ``compiled.memory_analysis()``: ``temp + arg + output −
        alias``.  XLA-visible buffers only.
    cufft_scratch
        Max plan workspace across the distinct FFT ops in the kernel
        (steady state — only one plan's scratch is live at any moment).
    total
        ``compiled_peak + cufft_scratch``.  Compare against your budget.
    fft_specs
        The :class:`FftSpec` instances parsed from the HLO; useful for
        diagnostic prints.
    """
    compiled_peak: int
    cufft_scratch: int
    total: int
    fft_specs: tuple[FftSpec, ...]


class HloFftParseError(RuntimeError):
    """Raised when the HLO FFT-op regex fails to match a recognized form.

    Loud-failure design choice: if XLA renames or reformats its FFT op
    text, we want to notice immediately.  Falling back silently to
    ``compiled_peak`` alone reintroduces the case-3 under-prediction
    (cuFFT scratch invisible) that motivated this module.
    """


class CufftQueryError(RuntimeError):
    """Raised when ``cufftMakePlanMany``/``cufftGetSize`` fails."""


# ---------------------------------------------------------------------------
# HLO FFT-op parser
# ---------------------------------------------------------------------------


# Matches the lowered HLO line for an XLA fft op.  JAX/XLA emits two
# HLO printout flavours; we accept both:
#
# (A) HLO-with-typed-operands (older / verbose):
#     %fft.1 = c128[10,75,75,200]{3,2,1,0} fft(c128[10,75,75,200]{3,2,1,0}
#         %arg.0), fft_type=FFT, fft_length={75,75,200}
#
# (B) HLO-without-typed-operands (current ``compiled.as_text()``):
#     ROOT %fft.0 = c64[10,60,60,80]{3,2,1,0} fft(%x.1),
#         fft_type=FFT, fft_length={60,60,80}, metadata={...}
#
# Both forms put the *output* dtype and shape on the LHS before ``=``,
# and that's what we parse.  For C2C ops (FFT/IFFT) output shape ==
# input shape, so the LHS gives us everything we need to build a cuFFT
# plan.  For R2C/C2R the half-spectrum convention means we'd need the
# operand shape too — see ``_resolve_real_fft_operand_shape``.
_FFT_OP_RE = re.compile(
    r"""
    %\S+ \s*=\s*                            # SSA result name
    (?P<dtype>[a-z]+\d+)                    # 'c128' / 'c64' / 'f64' / 'f32'
    \s* \[
    (?P<shape>[\d,\s]+)                     # output shape
    \]
    (?:\{[\d,\s]*\})?                       # optional layout
    \s*
    fft\s*\(                                # the 'fft(' call
    [^)]*                                   # operand expression (typed or not)
    \)
    \s*,\s* fft_type \s*=\s*
    (?P<fft_type>FFT|IFFT|RFFT|IRFFT)
    \s*,\s* fft_length \s*=\s* \{
    (?P<fft_length>[\d,\s]+)                # transform extents
    \}
    """,
    re.VERBOSE,
)


def parse_fft_specs_from_hlo(hlo_text: str) -> list[FftSpec]:
    """Walk an HLO text dump and return one :class:`FftSpec` per fft op.

    Empty list is a valid result (the kernel has no FFTs).  But if the
    text contains the substring ``" fft("`` and our regex doesn't match
    it, we raise :class:`HloFftParseError` — XLA changed format, the
    caller deserves to know.

    Parameters
    ----------
    hlo_text
        Output of ``compiled.as_text()`` (or any HLO dump).

    Returns
    -------
    list of FftSpec
    """
    specs: list[FftSpec] = []
    seen_spans: list[tuple[int, int]] = []
    for m in _FFT_OP_RE.finditer(hlo_text):
        seen_spans.append(m.span())
        dtype = m.group("dtype")
        op_shape = tuple(int(x) for x in m.group("shape").split(",") if x.strip())
        fft_length = tuple(int(x) for x in m.group("fft_length").split(",") if x.strip())
        rank = len(fft_length)
        if rank not in (1, 2, 3):
            raise HloFftParseError(
                f"Parsed fft_length={fft_length} has unsupported rank {rank}; "
                f"cuFFT supports rank 1/2/3 only."
            )
        # Trailing ``rank`` dims of the operand are the transform dims;
        # everything before them is batched (multiplicatively).
        if len(op_shape) < rank:
            raise HloFftParseError(
                f"Operand shape {op_shape} has fewer dims than fft rank {rank}; "
                f"check XLA HLO format — parser may need updating."
            )
        if op_shape[-rank:] != fft_length and m.group("fft_type") in ("FFT", "IFFT"):
            # For C2C the output shape's transform dims should match
            # fft_length exactly.  RFFT/IRFFT have a half-spectrum
            # convention (output's last transform dim is N/2+1), so
            # we don't enforce equality there — the cuFFT type code
            # accounts for the shape change.
            raise HloFftParseError(
                f"Output shape {op_shape} trailing dims do not match "
                f"fft_length={fft_length}; HLO format may have shifted."
            )
        batch = 1
        for d in op_shape[:-rank] if rank else op_shape:
            batch *= d
        specs.append(FftSpec(
            rank=rank,
            transform_shape=fft_length,
            batch=batch,
            dtype=dtype,
            fft_type=m.group("fft_type"),
        ))

    # Sanity: if the HLO mentions ' fft(' but our parser found nothing,
    # the regex needs updating.  Loud failure rather than silent zero.
    if " fft(" in hlo_text and not specs:
        raise HloFftParseError(
            "HLO text contains ' fft(' but the FFT-op regex matched zero "
            "ops.  XLA may have changed its lowered HLO format.  Update "
            "_FFT_OP_RE in src/runtime/aot_memory.py."
        )

    return specs


# ---------------------------------------------------------------------------
# libcufft handle (resolved via jaxlib's loaded shared object)
# ---------------------------------------------------------------------------


# cufftType codes.  Source: cuda/include/cufft.h.
_CUFFT_R2C = 0x2A   # single-precision real → complex
_CUFFT_C2R = 0x2C
_CUFFT_C2C = 0x29
_CUFFT_D2Z = 0x6A   # double-precision real → complex
_CUFFT_Z2D = 0x6C
_CUFFT_Z2Z = 0x69

# CUFFT_SUCCESS == 0; non-zero means error.
_CUFFT_SUCCESS = 0


def _cufft_type_for(dtype: str, fft_type: str) -> int:
    """Map (operand dtype, op kind) → cuFFT type code.

    ``c128`` / ``c64`` operands are complex; FFT / IFFT both use C2C
    (the direction is a runtime arg, not a plan property).  RFFT goes
    R2C, IRFFT goes C2R.  Half-precision is not supported.
    """
    if dtype == "c128":
        if fft_type in ("FFT", "IFFT"):
            return _CUFFT_Z2Z
        if fft_type == "RFFT":
            return _CUFFT_D2Z
        if fft_type == "IRFFT":
            return _CUFFT_Z2D
    elif dtype == "c64":
        if fft_type in ("FFT", "IFFT"):
            return _CUFFT_C2C
        if fft_type == "RFFT":
            return _CUFFT_R2C
        if fft_type == "IRFFT":
            return _CUFFT_C2R
    elif dtype == "f64" and fft_type == "RFFT":
        return _CUFFT_D2Z
    elif dtype == "f32" and fft_type == "RFFT":
        return _CUFFT_R2C
    raise CufftQueryError(
        f"Unsupported (dtype={dtype}, fft_type={fft_type}) combination."
    )


@functools.lru_cache(maxsize=1)
def _jax_cufft_handle() -> ctypes.CDLL:
    """Locate jaxlib's ``libcufft.so`` via ``/proc/self/maps`` and load it.

    Forces ``jax.devices()`` first so jaxlib has actually ``dlopen``'d
    libcufft.  We then read the process's mmap'd shared objects and
    pick the libcufft entry; that path is exactly what jaxlib loaded,
    so any cuFFT plan we build with this handle is bit-identical to
    the plan the runtime FFT thunk will build.

    Raises :class:`CufftQueryError` if libcufft is not present in the
    process map after CUDA init (CPU-only jax build, missing GPU
    backend).  Caller may catch this and skip the cuFFT contribution.
    """
    # Force CUDA platform init.  No-op if already done.
    try:
        jax.devices("gpu")
    except Exception as e:
        raise CufftQueryError(
            f"jax.devices('gpu') failed; cannot query cuFFT scratch: {e}"
        ) from e

    # /proc/self/maps lines look like:
    #   7f1234abc000-7f1234def000 r-xp 00000000 fd:01 12345     /path/to/libcufft.so.12
    # Path is the last whitespace-separated column (when present).
    cufft_path: str | None = None
    try:
        with open("/proc/self/maps", "r") as fh:
            for line in fh:
                parts = line.rstrip().split(None, 5)
                if len(parts) < 6:
                    continue
                path = parts[5]
                if not path.startswith("/"):
                    continue  # anonymous mapping like '[heap]', '[stack]'
                if os.path.basename(path).startswith("libcufft.so"):
                    cufft_path = path
                    break
    except OSError as e:
        raise CufftQueryError(
            f"could not read /proc/self/maps: {e}"
        ) from e

    if cufft_path is None:
        raise CufftQueryError(
            "libcufft.so not found in /proc/self/maps after jax.devices() "
            "— is this a CPU-only jax build?"
        )

    try:
        lib = ctypes.CDLL(cufft_path, mode=ctypes.RTLD_GLOBAL)
    except OSError as e:
        raise CufftQueryError(
            f"failed to ctypes-load {cufft_path}: {e}"
        ) from e

    # Bind ABI signatures.  See cuda/include/cufft.h.
    #   cufftResult cufftCreate(cufftHandle *plan)
    lib.cufftCreate.restype = ctypes.c_int
    lib.cufftCreate.argtypes = [ctypes.POINTER(ctypes.c_int)]

    #   cufftResult cufftDestroy(cufftHandle plan)
    lib.cufftDestroy.restype = ctypes.c_int
    lib.cufftDestroy.argtypes = [ctypes.c_int]

    #   cufftResult cufftSetAutoAllocation(cufftHandle plan, int autoAllocate)
    lib.cufftSetAutoAllocation.restype = ctypes.c_int
    lib.cufftSetAutoAllocation.argtypes = [ctypes.c_int, ctypes.c_int]

    #   cufftResult cufftMakePlanMany(plan, rank, n[], inembed[], istride,
    #                                 idist, onembed[], ostride, odist,
    #                                 type, batch, *workSize)
    # We pass NULL for inembed/onembed (default = packed contiguous).
    lib.cufftMakePlanMany.restype = ctypes.c_int
    lib.cufftMakePlanMany.argtypes = [
        ctypes.c_int,                        # plan
        ctypes.c_int,                        # rank
        ctypes.POINTER(ctypes.c_int),        # n
        ctypes.POINTER(ctypes.c_int),        # inembed (NULL)
        ctypes.c_int,                        # istride
        ctypes.c_int,                        # idist
        ctypes.POINTER(ctypes.c_int),        # onembed (NULL)
        ctypes.c_int,                        # ostride
        ctypes.c_int,                        # odist
        ctypes.c_int,                        # type
        ctypes.c_int,                        # batch
        ctypes.POINTER(ctypes.c_size_t),     # workSize
    ]

    return lib


# ---------------------------------------------------------------------------
# cuFFT scratch query
# ---------------------------------------------------------------------------


@functools.lru_cache(maxsize=512)
def _query_one_plan_workspace_bytes(spec: FftSpec) -> int:
    """Build a cuFFT plan with ``spec``'s parameters and return scratch.

    No GPU memory beyond the plan descriptor is allocated
    (``cufftSetAutoAllocation(plan, 0)``).  Cached by ``spec`` — same
    plan params return the same scratch value within a process.
    """
    lib = _jax_cufft_handle()

    plan = ctypes.c_int(0)
    rc = lib.cufftCreate(ctypes.byref(plan))
    if rc != _CUFFT_SUCCESS:
        raise CufftQueryError(f"cufftCreate failed: rc={rc}")

    try:
        rc = lib.cufftSetAutoAllocation(plan, 0)
        if rc != _CUFFT_SUCCESS:
            raise CufftQueryError(f"cufftSetAutoAllocation failed: rc={rc}")

        rank = spec.rank
        n_arr = (ctypes.c_int * rank)(*spec.transform_shape)

        # Verify int32 fits — cuFFT's plain (non-64) API is int-batch.
        # If batch >= 2**31, we'd need cufftMakePlanMany64; in practice
        # our V_q kernel batches stay well under that.
        if spec.batch >= (1 << 31):
            raise CufftQueryError(
                f"batch={spec.batch} exceeds int32 range; "
                f"use cufftMakePlanMany64 (not yet implemented here)."
            )

        ctype_code = _cufft_type_for(spec.dtype, spec.fft_type)

        # idist / odist must be the per-batch element count.  For C2C
        # this is product(n).  For R2C the output is half-spectrum
        # (n[-1]/2 + 1), and vice versa for C2R; jaxlib's FFT thunk
        # passes those distances explicitly (see fft_thunk.cc), so we
        # mirror the same convention.
        in_dist = 1
        out_dist = 1
        for ni in spec.transform_shape:
            in_dist *= int(ni)
            out_dist *= int(ni)
        if spec.fft_type == "RFFT":
            # output last dim is n[-1]/2 + 1
            out_dist = (out_dist // int(spec.transform_shape[-1])) * (
                int(spec.transform_shape[-1]) // 2 + 1)
        elif spec.fft_type == "IRFFT":
            in_dist = (in_dist // int(spec.transform_shape[-1])) * (
                int(spec.transform_shape[-1]) // 2 + 1)

        work_size = ctypes.c_size_t(0)
        rc = lib.cufftMakePlanMany(
            plan,                # cufftHandle
            rank,                # rank
            n_arr,               # n[]
            None,                # inembed (NULL → packed)
            1, in_dist,          # istride, idist
            None,                # onembed (NULL → packed)
            1, out_dist,         # ostride, odist
            ctype_code,          # type
            spec.batch,          # batch
            ctypes.byref(work_size),
        )
        if rc != _CUFFT_SUCCESS:
            raise CufftQueryError(
                f"cufftMakePlanMany failed: rc={rc} for spec={spec} "
                f"(idist={in_dist}, odist={out_dist}, type=0x{ctype_code:x})"
            )
        return int(work_size.value)
    finally:
        lib.cufftDestroy(plan)


def query_cufft_workspace_bytes(specs) -> int:
    """Sum cuFFT plan workspace across distinct ``FftSpec``s.

    "Distinct" means deduped by spec equality (same transform shape +
    batch + dtype + op kind ⇒ same plan ⇒ shared scratch).  Returns the
    **max** across distinct plans rather than the sum, since at runtime
    only one plan's scratch is live at a time within a single kernel
    (the FFT thunk creates the plan, executes, releases scratch — XLA
    schedules other thunks before/after but not concurrently).

    Returns 0 if ``specs`` is empty.
    """
    distinct = set(specs)
    if not distinct:
        return 0
    return max(_query_one_plan_workspace_bytes(s) for s in distinct)


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def aot_kernel_peak_bytes(compiled) -> AotPeakBreakdown:
    """Predict per-rank GPU peak for a JAX-compiled kernel including cuFFT scratch.

    Parameters
    ----------
    compiled
        Result of ``jax.jit(fn).lower(*specs).compile()`` (a
        ``jax.stages.Compiled``).

    Returns
    -------
    AotPeakBreakdown

    Notes
    -----
    On non-CUDA backends or when libcufft cannot be located, the cuFFT
    contribution is omitted and ``cufft_scratch == 0``.  The HLO parser
    still runs — if it fails to recognise an FFT op, we raise loudly
    (we want to notice format drift).
    """
    m = compiled.memory_analysis()
    compiled_peak = (
        int(m.temp_size_in_bytes)
        + int(m.argument_size_in_bytes)
        + int(m.output_size_in_bytes)
        - int(m.alias_size_in_bytes)
    )

    hlo_text = compiled.as_text()
    fft_specs = tuple(parse_fft_specs_from_hlo(hlo_text))

    # Only attempt the cuFFT query when there are FFTs to query and a
    # CUDA backend is available.  CufftQueryError on missing libcufft
    # demotes to ``cufft_scratch=0`` rather than failing the whole
    # prediction — chooser still works on CPU sandboxes.
    cufft_scratch = 0
    if fft_specs:
        try:
            cufft_scratch = query_cufft_workspace_bytes(fft_specs)
        except CufftQueryError:
            cufft_scratch = 0

    return AotPeakBreakdown(
        compiled_peak=compiled_peak,
        cufft_scratch=cufft_scratch,
        total=compiled_peak + cufft_scratch,
        fft_specs=fft_specs,
    )
