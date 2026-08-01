"""Flat-k batched 3-D FFT — the ``LORRAX_FFT_FFI`` / ``LORRAX_FFT_FFI_FUSED``
service.

The Python half of the flat-k FFT handlers.  ONE set of ``ffi_call`` sites
serves BOTH platforms, because the two libraries deliberately register the
SAME target strings under different C++ symbols
(``ffi_loader.py:99-107`` CUDA vs ``:140-142`` host — the phdf5
same-target/different-symbol split):

    cpu   liblorrax_ffi_host.so   MKL FFT via the DFTI *descriptor* API
                                  (``src/ffi/cpp/mklfft/fft_flat_k_ffi.cc``) —
                                  a genuine O(N log N) FFT at any k-count;
                                  NOT a DFT-as-matmul (owner-vetoed).
    CUDA  liblorrax_ffi.so        cuFFT with the ADVANCED DATA LAYOUT
                                  (``src/ffi/cpp/cufft/fft_flat_k_cuda_ffi.cc``):
                                  ``cufftPlanMany64`` inembed/istride=T/idist=1,
                                  the exact stride-descriptor analog; an
                                  NVRTC-compiled kernel fuses the G·W
                                  multiply + norms.

That is why ``src/ffi/cufft/`` has no Python module of its own — see its
``__init__.py``.  Full contract: ``docs/dev/flat_k_fft_service.md``.

WHY the service exists: XLA:CPU's ``fft`` custom-call requires the
transformed axes minor-most, so every ``dot`` (k-major flat) ↔ ``fft``
(k-minor 3-D) boundary in the Σ τ kernel pays a full transpose copy of the
~398 MB/rank μ² tile — measured 60-65% of ``sigma.exec`` at nb=128/P=64 and
CLOSED as structural for any XLA-side arrangement
(``wk_REL/sigma_perf_results.md``).  Stride descriptors read the dot-layout
tile where it lies, so the transposes disappear instead of moving.

Two entry LAYERS, and the gate reaches only one — stated because it is
structural, not a TODO.  ``make_flat_k_*`` wraps its own ``shard_map`` and
is FFI-gated; ``fft_helpers.local_fftn3``/``local_ifftn3`` are bare
``jnp.fft`` aliases for code ALREADY inside a ``shard_map`` (which cannot
nest) and have no FFI route at all.  ``isdf/core.py`` and
``common/wfn_transforms.py`` call the second layer, so ``LORRAX_FFT_FFI``
structurally cannot reach them.

ADOPTION STATE (2026-07-30; superseded 2026-07-31): this module IS the
single implementation.  ``common/fft_helpers.py`` delegated — it imports
the gate and both wrapper bodies from here (``fft_helpers.py:304``) and
carries no copy of its own.  The equivalence pin ``wk_REL/gatecheck.py``
(cells A2/E/E2) now guards the re-export seam rather than a second copy.
"""

from __future__ import annotations

import math
from typing import Callable

import jax
import jax.numpy as jnp
import numpy as np
from jax.experimental.shard_map import shard_map
from jax.sharding import Mesh, PartitionSpec as P

from ffi.gate import Gate

__all__ = [
    "FLAT_K_TARGET", "GW_CONV_TARGET", "GATE", "FUSED_GATE",
    "fft_ffi_enabled", "fft_ffi_mode",
    "fused_fft_ffi_enabled", "fused_fft_ffi_mode",
    "require_fft_ffi", "make_flat_k_fft_ffi", "make_gw_conv_ffi",
    "ffi_fft_scale", "validate_flat_spec",
]

FLAT_K_TARGET = "lorrax_mklfft_flat_k"
GW_CONV_TARGET = "lorrax_mklfft_gw_conv"

#: The ``LORRAX_FFT_FFI`` dial.  Default OFF and **no ``auto`` mode** — the
#: vocabulary is two-valued, so ``LORRAX_FFT_FFI=auto`` is a grammar error
#: here rather than a silently different policy from the GEMM dial.  Whether
#: this dial should GET an ``auto`` (the GPU margin is 2.1-2.4x at
#: production shapes, ``wk_REL/audit_gpu_fft.log:104-105``, ``:119-120`` —
#: larger than the margin that justified the GEMM auto) is an owner call;
#: the defect recorded by the 2026-07-30 assessment is the absence of a
#: recorded REASON, not the value.
GATE = Gate(
    env="LORRAX_FFT_FFI",
    target=FLAT_K_TARGET,
    platforms=("cpu", "CUDA"),
    modes=("off", "on"),
    default="off",
    off_label="default XLA FFT path",
    label={"cpu": "MKL FFT (DFTI API) host",
           "CUDA": "cuFFT strided CUDA"},
    resolved_msg={
        "cpu": ("[fft_ffi] flat-k 3-D FFTs -> MKL FFT (DFTI API) host "
                "FFI handler ({target}): O(N log N) FFT reading the "
                "dot-layout tile in place via stride descriptors — no "
                "XLA layout transposes."),
        "CUDA": ("[fft_ffi] flat-k 3-D FFTs -> cuFFT strided CUDA FFI "
                 "handler ({target}): cufftPlanMany64 advanced layout "
                 "(istride=T, idist=1 — the stride-descriptor analog) "
                 "reading the dot-layout tile in place; jnp.fft norm "
                 "scales applied by a fused device kernel."),
    },
    refuse_platform_msg=(
        "LORRAX_FFT_FFI requested the FFI flat-k FFT backend, but the mesh "
        "devices are '{platform}' — backends exist for cpu (MKL FFT via the "
        "DFTI API) and CUDA (cuFFT strided) only.  Unset LORRAX_FFT_FFI on "
        "this mesh (explicit requests are never silently downgraded)."),
    refuse_probe_msg=(
        "LORRAX_FFT_FFI requested the {label} "
        "backend, but FFI target '{target}' is unusable on platform "
        "'{platform}': {reason}"),
)

#: The ``LORRAX_FFT_FFI_FUSED`` dial — GRAMMAR ONLY, deliberately.
#:
#: This flag selects WHICH ENTRY POINT the τ kernel builds (one fused
#: ``gw_conv`` call vs the decomposed three-FFT chain), not which backend
#: serves it: the platform/handler refusal for ``lorrax_mklfft_gw_conv`` is
#: the FFT service's and is issued by ``GATE.require(target=GW_CONV_TARGET)``
#: inside :func:`make_gw_conv_ffi`, exactly as it is today.  Carrying a
#: second set of refusal prose here would create two wordings for one
#: condition.
#:
#: Until 2026-07-30 this flag was read at a CONSUMER
#: (``gw/ppm_tau_kernel.py:81``) with ``in ("1","true","yes","on")`` and no
#: grammar check, in violation of the service's own stated rule
#: (``fft_helpers.py:306-307``): ``=yes`` worked, ``=Y`` silently did
#: nothing, and neither printed anything.  Every spelling that worked before
#: still works; the only change is that an unrecognized value now SAYS so.
FUSED_GATE = Gate(
    env="LORRAX_FFT_FFI_FUSED",
    target=GW_CONV_TARGET,
    platforms=("cpu", "CUDA"),
    modes=("off", "on"),
    default="off",
    off_label="decomposed IFFT/FFT chain, production path untouched",
)


def fft_ffi_mode() -> str:
    """``"on"`` | ``"off"`` — the raw ``LORRAX_FFT_FFI`` grammar."""
    return GATE.mode()


def fft_ffi_enabled() -> bool:
    """True when ``make_flat_k_*`` should return the FFI variant.

    Read at helper-FACTORY time; kernel caches must key on it
    (``gw.ppm_tau_kernel``).  Backend-init-free (gate contract tier 1)."""
    return GATE.enabled()


def fused_fft_ffi_mode() -> str:
    """``"on"`` | ``"off"`` — the raw ``LORRAX_FFT_FFI_FUSED`` grammar."""
    return FUSED_GATE.mode()


def fused_fft_ffi_enabled() -> bool:
    """True when the τ kernel's IFFT·(G·W)·FFT step should be built as ONE
    fused FFI call (:func:`make_gw_conv_ffi`) instead of the decomposed
    three-transform chain.  Independent of ``LORRAX_FFT_FFI``; default OFF;
    read at kernel-factory time and part of the kernel cache keys."""
    return FUSED_GATE.enabled()


def require_fft_ffi(mesh: Mesh, target: str = FLAT_K_TARGET) -> str:
    """Announce-or-refuse for the requested FFI backend; returns the FFI
    platform key (``"cpu"`` / ``"CUDA"``).

    Refuses (with the ``probe_target`` reason) ONLY if the mesh platform has
    no backend, or the platform's library lacks the target — never silently
    runs the XLA path (refusal doctrine #8)."""
    return GATE.require(mesh, target=target)


def ffi_fft_scale(kind: str, norm: str | None, nk: int) -> float:
    """Total scale matching jnp.fft's norm conventions exactly:
    ifftn: backward/None -> 1/N, ortho -> 1/sqrt(N), forward -> 1;
    fftn : backward/None -> 1,  ortho -> 1/sqrt(N), forward -> 1/N.

    Computed HERE, in Python, and shipped to the handler as a plain scale —
    the handlers implement no norm convention of their own."""
    if norm == 'ortho':
        return 1.0 / math.sqrt(float(nk))
    if norm in (None, 'backward'):
        return 1.0 / float(nk) if kind == 'ifftn' else 1.0
    if norm == 'forward':
        return 1.0 if kind == 'ifftn' else 1.0 / float(nk)
    raise ValueError(f"Unsupported FFT norm={norm!r}")


def validate_flat_spec(spec: P, what: str) -> P:
    """FFT axes (leading three of the 3-D form) must be replicated; return
    the equivalent flat-form spec (nk axis replicated + original trail)."""
    axes = tuple(spec)
    if len(axes) < 3 or any(ax is not None for ax in axes[:3]):
        raise ValueError(
            f"FFI flat-k backend needs the three k axes of {what} replicated "
            f"(spec {spec}); sharded FFT axes are unsupported (same contract "
            f"as the XLA-path helpers).")
    return P(None, *axes[3:])


def make_flat_k_fft_ffi(
    mesh: Mesh,
    kgrid: tuple[int, int, int],
    spec: P,
    *,
    kind: str,
    norm: str | None,
    out_spec: P | None,
) -> Callable:
    """FFI-backed flat-k FFT: ``(nk, *trail) -> (nk, *trail)``, same contract
    as ``fft_helpers.make_flat_k_fft`` — one batched strided FFT per rank
    over the local shard (MKL DFTI on cpu, cuFFT advanced layout on CUDA),
    k-major layout end to end (never reshaped to the 3-D k-minor form, which
    is the whole point).

    FACTORY-time refusals: unsupported mesh platform, missing handler,
    ``out_spec`` reshard.  TRACE-time refusals: non-c128 dtype, rank, and
    leading extent — those are trace-time FACTS and cannot fire earlier
    (the two-phase contract; ``docs/dev/ffi_gate_contract.md``).

    ``input_output_aliases={0: 0}``: operand 0 is aliased to the result, so
    when the buffer is dead XLA lets the handler transform it in place — the
    terminal form of donation (zero extra big tiles).
    """
    if kind not in ('ifftn', 'fftn'):
        raise ValueError(f"kind must be 'ifftn' or 'fftn', got {kind!r}")
    if out_spec is not None and tuple(out_spec) != tuple(spec):
        raise ValueError(
            "FFI flat-k backend does not implement a post-FFT reshard "
            f"(out_spec {out_spec} != spec {spec}); unset LORRAX_FFT_FFI for "
            "this call path or drop out_spec.")
    require_fft_ffi(mesh, FLAT_K_TARGET)
    nkx, nky, nkz = (int(v) for v in kgrid)
    nk = nkx * nky * nkz
    flat_spec = validate_flat_spec(spec, "the input")
    scale = ffi_fft_scale(kind, norm, nk)
    attrs = dict(nkx=np.int64(nkx), nky=np.int64(nky), nkz=np.int64(nkz),
                 forward=np.int64(0 if kind == 'ifftn' else 1),
                 scale=np.float64(scale))

    def _local(x_local):
        out_t = jax.ShapeDtypeStruct(x_local.shape, x_local.dtype)
        return jax.ffi.ffi_call(
            FLAT_K_TARGET, out_t,
            input_output_aliases={0: 0},  # in-place when the operand is dead
        )(x_local, **attrs)

    _sm = shard_map(_local, mesh=mesh,
                    in_specs=(flat_spec,), out_specs=flat_spec,
                    check_rep=False)

    def _flat_k_fft_ffi(x_flat):
        if x_flat.dtype != jnp.complex128:
            raise TypeError(
                f"FFI flat-k backend supports complex128 only, got "
                f"{x_flat.dtype} (the XLA path would accept it — unset "
                f"LORRAX_FFT_FFI for this call path).")
        if x_flat.ndim != len(tuple(flat_spec)):
            raise ValueError(
                f"flat-k input rank {x_flat.ndim} does not match the "
                f"3-D-form spec {spec} (expect rank {len(tuple(flat_spec))} "
                f"flat).")
        if int(x_flat.shape[0]) != nk:
            raise ValueError(
                f"flat-k input leading extent {x_flat.shape[0]} != "
                f"nkx*nky*nkz = {nk}.")
        return _sm(x_flat)

    return _flat_k_fft_ffi


def make_gw_conv_ffi(
    mesh: Mesh,
    kgrid: tuple[int, int, int],
    g_spec: P,
    v_spec: P,
    *,
    norm: str | None = 'ortho',
    mult: float = 1.0,
) -> Callable:
    """FUSED flat-k convolution — the second FFI entry point (MKL DFTI on
    cpu, cuFFT + fused multiply kernel on CUDA).

    Returns ``fn(G_flat, W_flat) -> sigma_flat`` computing, value-identically
    to the decomposed helper sequence (~1e-15 rel; gated, not bit-exact)::

        sigma = fftn( ifftn(G) * ifftn(W)[:, None, :, None, :] * mult )

    with all three transforms + the broadcast multiply inside ONE FFI call
    per rank, chunked so the R-space G tile never materializes (the Σ τ
    kernel's big intermediate).  ``G_flat`` is ``(nk, a, mx, b, my)``,
    ``W_flat`` is ``(nk, mx, my)``; ``mult`` (e.g. Σ's -1/√N_k) is folded
    into the forward-transform scale.  Shapes/strides come from the runtime
    shards — nothing deck-specific.  Sigma-family layout contract only; the
    plain helpers remain the entry point for everything else.

    Refuses through the ``LORRAX_FFT_FFI`` gate's platform/handler guards
    (mode-independent — see :meth:`ffi.gate.Gate.require`): a caller
    that constructs this factory has already decided to use the handler, so
    "which flag is set" is not the question being asked here.
    """
    require_fft_ffi(mesh, GW_CONV_TARGET)
    nkx, nky, nkz = (int(v) for v in kgrid)
    nk = nkx * nky * nkz
    g_flat = validate_flat_spec(g_spec, "G")
    v_flat = validate_flat_spec(v_spec, "W")
    attrs = dict(nkx=np.int64(nkx), nky=np.int64(nky), nkz=np.int64(nkz),
                 scale_i=np.float64(ffi_fft_scale('ifftn', norm, nk)),
                 scale_f=np.float64(ffi_fft_scale('fftn', norm, nk)
                                    * float(mult)))

    def _local(g_local, w_local):
        if g_local.ndim != 5 or w_local.ndim != 3:
            raise ValueError(
                f"gw_conv expects local G (nk, a, mx, b, my) and W "
                f"(nk, mx, my); got {g_local.shape} / {w_local.shape}.")
        if (g_local.shape[0] != w_local.shape[0]
                or g_local.shape[2] != w_local.shape[1]
                or g_local.shape[4] != w_local.shape[2]):
            raise ValueError(
                f"gw_conv G/W shard shapes disagree: {g_local.shape} vs "
                f"{w_local.shape} (need G[0]==W[0], G[2]==W[1], G[4]==W[2]).")
        out_t = jax.ShapeDtypeStruct(g_local.shape, g_local.dtype)
        return jax.ffi.ffi_call(
            GW_CONV_TARGET, out_t,
            input_output_aliases={0: 0},  # sigma_k in G_k's buffer when dead
        )(g_local, w_local, **attrs)

    _sm = shard_map(_local, mesh=mesh,
                    in_specs=(g_flat, v_flat), out_specs=g_flat,
                    check_rep=False)

    def _gw_conv(G_flat, W_flat):
        if G_flat.dtype != jnp.complex128 or W_flat.dtype != jnp.complex128:
            raise TypeError("gw_conv supports complex128 only.")
        if int(G_flat.shape[0]) != nk or int(W_flat.shape[0]) != nk:
            raise ValueError(
                f"gw_conv leading extents {G_flat.shape[0]}/{W_flat.shape[0]} "
                f"!= nkx*nky*nkz = {nk}.")
        return _sm(G_flat, W_flat)

    return _gw_conv
