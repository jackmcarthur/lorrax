"""Batched 3-D FFT and the FUSED-CONV FAMILY — the ``LORRAX_FFT_FFI`` /
``LORRAX_FFT_FFI_FUSED`` / ``LORRAX_CONV_KMINOR_FFI`` services.

The Python half of the flat-k FFT handlers.  ONE set of ``ffi_call`` sites
serves BOTH platforms, because the two libraries deliberately register the
SAME target strings under different C++ symbols
(``ffi_loader.py:99-107`` CUDA vs ``:140-142`` host — the phdf5
same-target/different-symbol split):

    cpu   liblorrax_ffi_host.so   the FFTW3 ABI (``fftw_plan_many_dft``, the
                                  advanced-layout planner) in
                                  (``src/ffi/cpp/mklfft/fft_flat_k_ffi.cc``) —
                                  a genuine O(N log N) FFT at any k-count;
                                  NOT a DFT-as-matmul (owner-vetoed).  The
                                  directory is still named ``mklfft`` for the
                                  DFTI implementation it USED to hold; the
                                  DFTI calls were deleted 2026-08-05 and the
                                  library is now bound by ``dlsym`` over a
                                  candidate ladder (``LORRAX_FFTW3_SO``).
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
~398 MB/rank μ² tile — measured 65% of the STAGED τ DISPATCH (191.9 s of
295.0 s) at nb=128/P=64 and CLOSED as structural for any XLA-side arrangement
(``wk_REL/sigma_perf_results.md``).  Stride descriptors read the dot-layout
tile where it lies, so the transposes disappear instead of moving.

MIND THE DENOMINATOR, and mind the tense.  These lines said "60-65% of
``sigma.exec``" until 2026-08-11; 191.9 s is 65% of the staged τ dispatch and
70.5% of ``sigma.exec`` (272.0 s), so the quoted range belonged to neither
(``wk_REL/FFI_EVIDENCE_AUDIT.md`` F26).  More importantly it is the number
from BEFORE this service existed, and reading it as current is how a lane
concludes the τ kernel is still FFT-bound and proposes wiring in the FFI that
is already wired.

MIND THE DECK TOO.  There is no single "after" number, because the FFT's share
is governed by K-POINT COUNT and every figure in the record was taken at small
nk.  Measured 2026-08-11 at P=4 on A100s, BFC@0.85, HEAD dc766220, as a share
of the staged τ dispatch: 16.1% on the 9-k gnppm_debug fixture, 60.5% at
Si 4x4x4 (64 k), and 84.9% at Si 6x6x6 (216 k), where the FFT is about 28% of
the whole driver wall.  The cpu nb=128/P=64 figures (15.1% decomposed / 7.6%
fused, F25) are a 64-k-class shape.  Cost goes as
n_tau * nk * mu_local * N_grid log N_grid.  Quote the rung or quote none —
a lane that took the fixture's 0.07%-of-wall as general concluded there was no
lever on the same day the correction landed
(``tests/known_failures/2026-08-11-gnppm-sigma-performance-claims-adjudicated.md``).

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

================================================================================
THE FUSED-CONV FAMILY — one contract, two resident layouts
================================================================================
Beyond the plain transform, this module owns a FAMILY of fused convolution
entries.  Every member computes the same thing::

    U = scale · FFT_k( IFFT_k(X) · K )

— one custom call for both transforms, the broadcast multiply against a
STORED kernel ``K``, and every norm factor folded into ONE constant, so the
R-space intermediate never materialises and the platform's missing-'ortho'
scale passes are never emitted.  What distinguishes the members is nothing
about the mathematics: it is **where the caller's tile already keeps its k
axis**, because that decides which memory layout the one kernel must read.

    member      k axis in X   handler target              platforms  factory
    ----------  ------------  --------------------------  ---------  --------
    k-strided   LEADING       lorrax_mklfft_gw_conv       cpu, CUDA  make_gw_conv_ffi
    k-minor     MINOR-most    lorrax_cufft_conv_kminor    CUDA       make_conv_kminor_ffi

THE CHOICE IS THE CALLER'S RESIDENT LAYOUT, AND IT IS MEASURED, NOT A TASTE.
The k-strided member reads a k-major tile through cuFFT's advanced data
layout (``cufftPlanMany64`` istride=T, idist=1), which is exactly right for
the Σ τ kernel: its ``dot`` layout is k-major, so the handler REMOVES a
transpose that would otherwise be paid on both sides of every transform.
Handed a tile that is already k-minor it is the wrong engine, and by a
growing margin — the strided plan's cost scales with the batch stride, and
for a caller whose stride is a full μ·ν tile that is measured at 1.61× the
XLA chain at nk=64 and 4.00× at nk=216 (2026-08-16,
``reports/screening_diagrams_wbse/evidence/opt_fftffi/``).  The k-minor
member exists for that caller: it reads the contiguous k-minor tile where it
lies, with a DIRECT per-axis DFT — no radix, no plan, and therefore no batch
stride to degrade — and it can emit a chosen output PERMUTATION from the store
so a downstream consumer's layout costs nothing extra.

So: pick the member whose k position matches the tile you already hold.
Neither is a fallback for the other, and a caller should never transpose to
reach one — transposing to reach a fused kernel spends exactly what the
fusion saves.  New members belong here, beside these two, under the same
contract.
"""

from __future__ import annotations

import math
from typing import Callable

import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import Mesh, PartitionSpec as P

from ffi.gate import Gate

# ``from common.shard_map import shard_map`` is DELIBERATELY NOT HERE.
# Importing ANY ``common`` submodule runs ``common/__init__.py``, which
# imports ``.wfn_transforms`` -> ``common.fft_helpers`` -> ``ffi.mklfft``
# -> ``from ffi.fft import FLAT_K_TARGET``.  Entering that cycle at THIS
# module means the re-entry finds ffi.fft half-executed and every name
# below still unbound:
#
#     ImportError: cannot import name 'FLAT_K_TARGET' from partially
#     initialized module 'ffi.fft' (most likely due to a circular import)
#
# so ``import ffi.fft`` was simply impossible as a process's first LORRAX
# import.  pytest never saw it (conftest enters at ``common`` first, where
# the cycle closes harmlessly), which is why it survived: the only thing
# that entered at ffi.fft was src/ffi/cpp/gate_one_fftw.sh's dynamic leg,
# and that gate has been unable to run since the cycle appeared.
#
# The import moves into the two factory bodies that use it (:277, :358) --
# the function-local spelling twelve other sites in this tree already use
# for common.shard_map, so this is the tree's own convention, not a
# workaround.  ffi.fft is L3 substrate; the ``common`` package __init__ is
# a physics-adjacent aggregator, and pulling all of it in to reach one
# version shim was the actual defect.

__all__ = [
    "FLAT_K_TARGET", "GW_CONV_TARGET", "GATE", "FUSED_GATE",
    "fft_ffi_enabled", "fft_ffi_mode",
    "fused_fft_ffi_enabled", "fused_fft_ffi_mode",
    "require_fft_ffi", "make_flat_k_fft_ffi", "make_gw_conv_ffi",
    "ffi_fft_scale", "validate_flat_spec",
    # the fused-conv family's k-MINOR member (see the module docstring)
    "CONV_KMINOR_TARGET", "CONV_KMINOR_GATE",
    "conv_kminor_mode", "conv_kminor_enabled", "require_conv_kminor",
    "conv_kminor_available", "conv_kminor_scale", "make_conv_kminor_ffi",
    "conv_kminor_plan", "conv_kminor_row_fits",
    "conv_kminor_out_shape", "conv_kminor_out_spec",
]

FLAT_K_TARGET = "lorrax_mklfft_flat_k"
#: The fused-conv family's k-STRIDED member.  ONE target string for both
#: platforms (the host and CUDA libraries register it under different C++
#: symbols) — the name is historical, coined by the CPU prototype.
GW_CONV_TARGET = "lorrax_mklfft_gw_conv"
#: The fused-conv family's k-MINOR member.  CUDA-ONLY, and named for the
#: vendor leg that carries it rather than borrowing the ``mklfft`` prefix:
#: there is no host twin, so a shared string would promise a cpu handler that
#: does not exist and a cpu mesh would resolve to nothing instead of refusing.
CONV_KMINOR_TARGET = "lorrax_cufft_conv_kminor"

#: The ``LORRAX_FFT_FFI`` dial.  Default ON — the FFI layer is REQUIRED
#: (owner ruling, ``docs/architecture/decisions.md`` 2026-08-01): the flat-k
#: XLA duplicate inside ``fft_helpers.make_flat_k_fft`` was deleted under
#: that ruling, so ``=0`` REFUSES (off_policy="refuse") instead of selecting
#: a path that no longer exists, and a missing/unloadable library is a
#: startup refusal naming the ``.so`` (``Gate.enforce``, wired into
#: ``runtime.initialize_communicator_stack``).  No ``auto`` mode — there is
#: nothing to auto-detect when the backend is mandatory.
GATE = Gate(
    env="LORRAX_FFT_FFI",
    target=FLAT_K_TARGET,
    platforms=("cpu", "CUDA"),
    modes=("off", "on"),
    default="on",
    off_label="(deleted) XLA flat-k FFT path",
    off_policy="refuse",
    off_refuse_msg=(
        "LORRAX_FFT_FFI=0: there is nothing to opt out to.  The XLA flat-k "
        "FFT path (the native-JAX duplicate inside "
        "common.fft_helpers.make_flat_k_fft) was DELETED under the "
        "FFI-required ruling (docs/architecture/decisions.md, 2026-08-01) — "
        "the certified backend is the platform FFI handler (the FFTW3 ABI "
        "on cpu, cuFFT strided on CUDA).  Unset LORRAX_FFT_FFI, or recover the "
        "XLA arm from git history for a debugging build.  BSE and the "
        "shard_map-interior local_*fftn3 aliases are unaffected (they never "
        "had an FFI route)."),
    # NOTE the platform names below are ABIs, not products.  The host handler
    # calls the FFTW3 ABI (`fftw_plan_many_dft` ×4 in
    # cpp/mklfft/fft_flat_k_ffi.cc; zero `DftiCreateDescriptor` since
    # 2026-08-05) and binds it by dlsym against whatever the process links —
    # cray-fftw, a system FFTW3, or MKL's FFTW3 wrappers via `libmkl_rt.so`.
    # These strings said "MKL FFT (DFTI API)" for the five days after the
    # DFTI code was deleted, so every CPU startup block named an engine the
    # translation unit no longer contained.  Name the ABI; let
    # LORRAX_FFT_FFI_LOG name the library.
    label={"cpu": "FFTW3-ABI host",
           "CUDA": "cuFFT strided CUDA"},
    resolved_msg={
        "cpu": ("[fft_ffi] flat-k 3-D FFTs -> FFTW3-ABI host FFI handler "
                "({target}): O(N log N) FFT reading the dot-layout tile "
                "in place via advanced-layout plans — no XLA layout "
                "transposes.  WHICH library answers is resolved at run "
                "time by dlsym over the candidate ladder (see "
                "LORRAX_FFTW3_SO and docs/architecture/ffi_layout.md §3), "
                "and is NOT stated by this line."),
        "CUDA": ("[fft_ffi] flat-k 3-D FFTs -> cuFFT strided CUDA FFI "
                 "handler ({target}): cufftPlanMany64 advanced layout "
                 "(istride=T, idist=1 — the stride-descriptor analog) "
                 "reading the dot-layout tile in place; jnp.fft norm "
                 "scales applied by a fused device kernel."),
    },
    refuse_platform_msg=(
        "LORRAX_FFT_FFI: the required FFI flat-k FFT backend cannot serve "
        "this mesh — its devices are '{platform}', and backends exist for "
        "cpu (the FFTW3 ABI) and CUDA (cuFFT strided) only."),
    refuse_probe_msg=(
        "The required {label} backend is unavailable: FFI target "
        "'{target}' is unusable on platform '{platform}': {reason}  The "
        "FFI layer is REQUIRED (docs/architecture/decisions.md, "
        "2026-08-01); build/locate the library per "
        "docs/environment/overview.md (host: build_host.sh -> "
        "liblorrax_ffi_host.so, selected by LORRAX_FFI_HOST_SO; CUDA: "
        "build.sh -> liblorrax_ffi.so, LORRAX_FFI_SO)."),
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
#: Default ON since the FFI-required ruling (decisions.md 2026-08-01): the
#: fused entry is the certified production form (GATES.md: certified ``on``
#: together with ``LORRAX_FFT_FFI``).  ``=0`` is a real opt-out
#: (off_policy="fallback"), NOT a refusal: the thing it selects — the
#: decomposed three-transform chain — is itself FFI-served through the same
#: required handlers, so it is a structural choice between two certified
#: FFI paths, not a native-JAX duplicate.
#:
#: Until 2026-07-30 this flag was read at a CONSUMER
#: (``gw/ppm_tau_kernel.py:81``) with ``in ("1","true","yes","on")`` and no
#: grammar check, in violation of the service's own stated rule: ``=yes``
#: worked, ``=Y`` silently did nothing, and neither printed anything.
#: Every spelling that worked before still works; the only change is that
#: an unrecognized value now SAYS so.
FUSED_GATE = Gate(
    env="LORRAX_FFT_FFI_FUSED",
    target=GW_CONV_TARGET,
    platforms=("cpu", "CUDA"),
    modes=("off", "on"),
    default="on",
    off_label="decomposed three-FFT chain (still FFI-served)",
    off_policy="fallback",
    off_announce_msg=(
        "[LORRAX_FFT_FFI_FUSED] =0: explicit opt-out — the tau kernel "
        "builds the decomposed IFFT/multiply/FFT chain instead of the "
        "fused gw_conv entry.  Both forms ride the required FFI handlers; "
        "the fused entry is the certified production default "
        "(decisions.md 2026-08-01)."),
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
    three-transform chain.  Independent of ``LORRAX_FFT_FFI``; default ON
    (decisions.md 2026-08-01); read at kernel-factory time and part of the
    kernel cache keys."""
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
    over the local shard (the FFTW3 ABI on cpu, cuFFT advanced layout on CUDA),
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

    from common.shard_map import shard_map     # see the import-cycle note
    _sm = shard_map(_local, mesh=mesh,
                    in_specs=(flat_spec,), out_specs=flat_spec,
                    check_vma=False)

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


# ===========================================================================
# THE FUSED-CONV FAMILY, member 1 of 2: k-STRIDED
# ===========================================================================
def make_gw_conv_ffi(
    mesh: Mesh,
    kgrid: tuple[int, int, int],
    g_spec: P,
    v_spec: P,
    *,
    norm: str | None = 'ortho',
    mult: float = 1.0,
) -> Callable:
    """FUSED flat-k convolution, **k-LEADING** — the family's k-strided member
    (the FFTW3 ABI on cpu, cuFFT + fused multiply kernel on CUDA).

    Its sibling is :func:`make_conv_kminor_ffi`, which computes the same
    expression for a caller whose tile keeps k MINOR-most; see the module
    docstring's family table for which one a given caller wants and why the
    choice is a measurement rather than a preference.

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

    from common.shard_map import shard_map     # see the import-cycle note
    _sm = shard_map(_local, mesh=mesh,
                    in_specs=(g_flat, v_flat), out_specs=g_flat,
                    check_vma=False)

    def _gw_conv(G_flat, W_flat):
        if G_flat.dtype != jnp.complex128 or W_flat.dtype != jnp.complex128:
            raise TypeError("gw_conv supports complex128 only.")
        if int(G_flat.shape[0]) != nk or int(W_flat.shape[0]) != nk:
            raise ValueError(
                f"gw_conv leading extents {G_flat.shape[0]}/{W_flat.shape[0]} "
                f"!= nkx*nky*nkz = {nk}.")
        return _sm(G_flat, W_flat)

    return _gw_conv


# ===========================================================================
# THE FUSED-CONV FAMILY, member 2 of 2: k-MINOR
# ===========================================================================
# The k-strided member above reads a k-LEADING tile through cuFFT's advanced
# data layout.  This one reads a k-MINOR tile — the layout a caller holds when
# its k axis is already innermost — with a DIRECT per-axis DFT against a
# runtime-built twiddle ring and no plan, so there is no batch stride to
# degrade, and it can emit a chosen output PERMUTATION straight from the store.
#
# SIZE-AGNOSTIC: the transform length of each axis is a call ATTRIBUTE, not a
# template parameter and not a compiled specialisation, so one code path serves
# every (nkx,nky,nkz) — primes and mixed radices included.  The only bound is
# RESIDENCY (the fused chain keeps the whole k-row live between its halves, so
# a row must fit a block's shared memory); it is derived from the device at run
# time and a k-grid over it is REFUSED BY NAME, quoting the device maximum and
# naming the k-strided member as the alternative.  Handler:
# ``src/ffi/cpp/cufft/conv_kminor_cuda_ffi.cc``.
#
# THE CONTRACT (generic; no caller is privileged):
#     X : (d0, d1, d2, d3, d4, nk) c128, contiguous, nk MINOR-most, replicated
#     K : (d1, d2, nk)             c128, ALREADY in R space, broadcast over
#                                  d0/d3/d4
#     U : out_layout=0 → shape(X), aliased to operand 0 (runs in place)
#         out_layout=1 → (d0, nk, d3, d1, d4, d2), emitted by the STORE
#
# "an ifft·multiply·fft against a stored kernel over a designated axis" — the
# five leading axes are free names.  A consumer with fewer than five folds its
# free axes into d0/d3/d4; a consumer with a different downstream layout adds
# an out_layout, it does not add a transpose.
#
# K IS NOT TRANSFORMED HERE, deliberately.  The stored kernel is the thing a
# caller builds ONCE (the BSE's ``bse_feast.ensure_W_R`` caches
# W_R = ifftn(W_q, norm='ortho') per solve; the tile is 22.9 MB at the gnppm
# fixture) and reuses across every application.  Transforming it inside the
# handler would repeat that transform on every call to save a call the caller
# already made — the opposite trade from the k-strided member, whose Σ caller
# has no such cache.  Same family, different amortisation; say which, do not
# split the difference.

#: The ``LORRAX_CONV_KMINOR_FFI`` dial — the family's k-minor member.
#:
#: **DEFAULT ``auto``** since the P>1 certification (2026-08-16).  Three modes,
#: and the middle one is the point:
#:
#:   ``auto`` (default) — use the kernel when the mesh is CUDA, the loaded
#:       device library exports the handler, AND the k-grid's row fits a
#:       block's shared memory.  Otherwise fall through to the caller's XLA
#:       chain, SILENTLY and CORRECTLY.  The silence is declared, not
#:       accidental: the fallthrough is the CERTIFIED REFERENCE, not a
#:       degraded twin, so there is nothing for a per-call warning to warn
#:       about.  Exactly one line, in the startup report, says which arm the
#:       run took.
#:   ``on``  — require it; refuse by name (naming the ``.so`` and the rebuild)
#:       if the platform or the handler cannot serve it.  For certification
#:       runs that must not silently measure the other arm.
#:   ``off`` — never; the XLA chain everywhere.
#:
#: WHY ``auto`` IS LEGITIMATE HERE and was deleted elsewhere: the dials it was
#: removed from are REQUIRED layers, where auto demoted onto a duplicate
#: compute path.  This dial's OFF state is the production implementation on
#: every backend, so ``auto`` selects an ACCELERATOR when one is present
#: rather than demoting when one is missing.  A CPU/ROCm/TPU mesh takes the
#: XLA arm by construction — that is the "NVIDIA GPU backend only" safety, and
#: it is a platform fact, not a runtime check that could go wrong.
#:
#: Read at FACTORY time, so the MODE (not a bool) is in ``ffi.ffi_dial_key``
#: and the variable is in ``common.jax_compile_cache.RANK_FINGERPRINT_ENV``:
#: it replaces four ops with one custom call, so two ranks disagreeing compile
#: modules with different op sets.
CONV_KMINOR_GATE = Gate(
    env="LORRAX_CONV_KMINOR_FFI",
    target=CONV_KMINOR_TARGET,
    platforms=("CUDA",),
    modes=("off", "auto", "on"),
    default="auto",
    off_label="the caller's XLA ifft/multiply/fft chain",
    off_policy="fallback",
    auto_capability=(
        "the mesh is CUDA, the loaded device library exports "
        "CufftConvKMinorCudaFfi, and the k-grid's row fits a block's shared "
        "memory"),
    auto_on_msg=(
        "[conv_kminor] auto -> ON: the fused k-minor conv kernel ({target}) "
        "serves this run's rung.  CUDA mesh, handler present.  Callers whose "
        "k-grid is too large for one resident k-row still take the XLA chain "
        "for that call, silently and correctly."),
    auto_off_msg=(
        "[conv_kminor] auto -> OFF: callers keep the XLA "
        "ifft/multiply/fft chain, which is the certified path on every "
        "backend.  Reason: {reason}"),
    off_announce_msg=(
        "[LORRAX_CONV_KMINOR_FFI] =0: the fused-conv family's k-minor member "
        "is disabled by request; callers keep the XLA ifft/multiply/fft "
        "chain.  The default is `auto`, which uses the kernel where it is "
        "available and falls through where it is not."),
    label={"CUDA": "k-minor fused conv CUDA"},
    resolved_msg={
        "CUDA": ("[conv_kminor] ifft·multiply·fft over the MINOR k axis -> "
                 "fused CUDA FFI handler ({target}): one kernel, one read of "
                 "the tile, one write, both norm factors folded into a "
                 "single constant, and out_layout=1 emits the consumer's "
                 "permuted layout from the store.  Direct per-axis DFT at "
                 "the k-grid the call names — no radix specialisation, no "
                 "plan, so no batch stride to degrade."),
    },
    refuse_platform_msg=(
        "LORRAX_CONV_KMINOR_FFI=1 requires the fused-conv family's k-minor "
        "member, which is CUDA-only, and this mesh's devices are "
        "'{platform}'.  It is a CUDA kernel, not a library call, so there is "
        "no host twin to demote to — unlike the k-strided member "
        "(lorrax_mklfft_gw_conv), which both platforms serve.  Use the "
        "default `auto` (it falls through to the XLA chain here), or unset "
        "the dial."),
    refuse_probe_msg=(
        "LORRAX_CONV_KMINOR_FFI=1 requested the {label} backend, but FFI "
        "target '{target}' is unusable on platform '{platform}': {reason}  "
        "This handler is NEW (2026-08-16, "
        "src/ffi/cpp/cufft/conv_kminor_cuda_ffi.cc): a device library built "
        "before it exists loads fine and simply does not export "
        "CufftConvKMinorCudaFfi.  Rebuild the CUDA leg "
        "(src/ffi/cpp/build.sh) and point LORRAX_FFI_SO at the result, or "
        "use the default `auto`, which falls through to the XLA chain."),
)


def conv_kminor_mode() -> str:
    """``"on"`` | ``"off"`` — the raw ``LORRAX_CONV_KMINOR_FFI`` grammar."""
    return CONV_KMINOR_GATE.mode()


def conv_kminor_enabled() -> bool:
    """True unless the dial is explicitly ``off``.

    Tier 1 (lexical): env only, no backend init, legal in a kernel cache key.
    It says nothing about whether the kernel will actually RUN — under ``auto``
    that is a mesh-and-shape question, answered by :func:`conv_kminor_plan`."""
    return CONV_KMINOR_GATE.enabled()


#: The shared-memory a launch may assume WITHOUT the device opt-in.  The
#: handler raises its own ceiling to the device maximum (queried, then
#: ``cuFuncSetAttribute``), but Python cannot see that number without a device
#: query of its own, and ``auto`` must not guess high: a guess that is too
#: generous turns a silent fallthrough into a mid-run refusal.  So ``auto``
#: uses the floor every CUDA device provides, and ``on`` lets the handler's own
#: derived bound decide — which is the mode a caller picks precisely when it
#: wants the real limit and a refusal if it is exceeded.
_CONV_KMINOR_AUTO_SMEM_FLOOR = 49152


def conv_kminor_row_fits(kgrid, smem_bytes: int = _CONV_KMINOR_AUTO_SMEM_FLOOR
                         ) -> bool:
    """Does ONE k-row of this grid fit ``smem_bytes`` of shared memory?

    The residency bound, mirrored from the handler's ``plan_launch`` so the
    Python side can answer it without a device round trip: the tile row is
    padded to an ODD element stride, and the twiddle rings and the one-row
    metadata slot share the block's allocation.
    """
    nkx, nky, nkz = (int(v) for v in kgrid)
    nk = nkx * nky * nkz
    return 16 * ((nk | 1) + 1 + nkx + nky + nkz) <= int(smem_bytes)


def conv_kminor_plan(mesh: Mesh, kgrid) -> tuple[bool, str]:
    """``(use_kernel, reason)`` — THE routing decision, in one place.

    This is what a consumer calls.  It folds the three-mode grammar, the
    platform, the handler probe and the residency bound into one answer, so no
    caller re-implements any part of the policy:

    * ``off``  → ``(False, ...)``, always.
    * ``on``   → ``require_conv_kminor`` (RAISES, naming the fix, if the
      platform or the handler cannot serve it), then the residency bound,
      which also raises rather than silently falling through: a caller that
      said ``on`` asked not to be routed elsewhere without being told.
    * ``auto`` → ``(True, ...)`` when CUDA + handler + the row fits;
      ``(False, reason)`` otherwise, and the caller takes its own path.  No
      exception, no per-call output — see the dial docstring for why the
      silence is declared rather than sloppy.
    """
    mode = CONV_KMINOR_GATE.mode()
    if mode == "off":
        return False, "LORRAX_CONV_KMINOR_FFI=off"
    fits = conv_kminor_row_fits(kgrid)
    if mode == "on":
        require_conv_kminor(mesh)          # raises with the fix named
        if not fits:
            nk = int(np.prod([int(v) for v in kgrid]))
            raise RuntimeError(
                f"LORRAX_CONV_KMINOR_FFI=on, but this call's k-grid "
                f"{tuple(int(v) for v in kgrid)} (nk={nk}) needs more shared "
                f"memory for ONE k-row than the {_CONV_KMINOR_AUTO_SMEM_FLOOR}"
                f" B every CUDA device guarantees.  The handler may still "
                f"serve it — it raises its ceiling to the DEVICE maximum and "
                f"reports the real bound — so either drop to the default "
                f"`auto` (which falls through to the XLA chain here) or call "
                f"the handler directly and read its refusal, which quotes "
                f"this device's own maximum.")
        return True, "on"
    ok, why = conv_kminor_available(mesh)
    if not ok:
        return False, why
    if not fits:
        return False, (
            f"k-grid {tuple(int(v) for v in kgrid)} needs more than "
            f"{_CONV_KMINOR_AUTO_SMEM_FLOOR} B of shared memory for one "
            f"k-row")
    return True, "auto"


def require_conv_kminor(mesh: Mesh) -> str:
    """Announce-or-REFUSE; returns the FFI platform key (``"CUDA"``).

    Mode-independent, like every other ``Gate.require``: a caller that has
    built this factory has already decided, so the only question here is
    whether this mesh can serve the handler."""
    return CONV_KMINOR_GATE.require(mesh, target=CONV_KMINOR_TARGET)


def conv_kminor_available(mesh: Mesh) -> tuple[bool, str]:
    """``(ok, reason)`` — the non-raising twin of :func:`require_conv_kminor`.

    For callers that must CHOOSE and report: a bench that wants "handler
    absent" as a row rather than a traceback, an opt-in hook that wants to
    print why it stayed off.  Anything already committed to the fused path
    calls :func:`require_conv_kminor` instead — silently selecting the other
    arm is the demotion the gate doctrine forbids."""
    try:
        require_conv_kminor(mesh)
    except Exception as exc:                       # noqa: BLE001 — reported
        return False, f"{type(exc).__name__}: {exc}"
    return True, "CUDA"


def conv_kminor_scale(norm: str | None, nk: int, mult: float = 1.0) -> float:
    """The ONE constant the handler applies: ``ifft norm · fft norm · mult``.

    Folding both ``jnp.fft`` norm factors into a single multiply is what
    deletes the pair of scale passes a cuFFT-backed chain emits (271 µs per
    ladder matvec at the gnppm fixture).  Computed HERE, in Python, exactly as
    :func:`ffi_fft_scale` is for the other members: the handlers implement no
    norm convention of their own."""
    return (ffi_fft_scale('ifftn', norm, nk)
            * ffi_fft_scale('fftn', norm, nk)
            * float(mult))


def conv_kminor_out_shape(x_shape, out_layout: int) -> tuple[int, ...]:
    """Output shape for an operand of shape ``x_shape`` at this ``out_layout``."""
    d0, d1, d2, d3, d4, nk = (int(v) for v in x_shape)
    if out_layout == 0:
        return (d0, d1, d2, d3, d4, nk)
    if out_layout == 1:
        return (d0, nk, d3, d1, d4, d2)
    raise ValueError(f"out_layout must be 0 or 1, got {out_layout!r}")


def conv_kminor_out_spec(x_spec: P, out_layout: int) -> P:
    """The output ``PartitionSpec`` INDUCED by ``x_spec`` at this layout.

    Derived, never passed: the permutation is a pure axis reorder, so each
    logical axis carries its mesh axis with it, and a caller that supplied its
    own spec could only supply one that disagreed."""
    ax = tuple(x_spec)
    if len(ax) != 6:
        raise ValueError(
            f"the k-minor conv contract is rank 6 (d0,d1,d2,d3,d4,nk); got "
            f"spec {x_spec} of rank {len(ax)}.")
    if out_layout == 0:
        return P(*ax)
    if out_layout == 1:
        return P(ax[0], ax[5], ax[3], ax[1], ax[4], ax[2])
    raise ValueError(f"out_layout must be 0 or 1, got {out_layout!r}")


def make_conv_kminor_ffi(
    mesh: Mesh,
    kgrid: tuple[int, int, int],
    x_spec: P,
    k_spec: P,
    *,
    norm: str | None = 'ortho',
    mult: float = 1.0,
    out_layout: int = 0,
) -> Callable:
    """FUSED convolution, **k-MINOR** — the family's k-minor member.

    Returns ``fn(X, K) -> U`` computing, value-identically to the decomposed
    chain up to reassociation (~1e-15 rel; gated, not bit-exact)::

        U = scale · fftn_k( ifftn_k(X) · K[None, :, :, None, None, :] )

    in ONE FFI call per rank, with ``scale = ifft-norm · fft-norm · mult``
    applied once.

    Parameters
    ----------
    x_spec, k_spec
        Rank-6 ``PartitionSpec`` of ``X`` ``(d0,d1,d2,d3,d4,nk)`` with the k
        axis REPLICATED, and rank-3 spec of ``K`` ``(d1,d2,nk)`` placing
        ``d1``/``d2`` on the same mesh axes.  The handler multiplies
        rank-local tiles and implements no reshard — the same contract every
        member of this family carries.
    kgrid
        ``(nkx, nky, nkz)``, product equal to the minor extent.  The 3-D
        structure exists only inside the kernel.
    norm, mult
        The two transforms' norm convention and any caller multiplier, folded
        into one constant.  ``K`` is ALREADY in R space (see the section
        header): ``norm`` describes what the HANDLER does, not how ``K`` was
        built.
    out_layout
        ``0`` — ``shape(X)``, aliased to operand 0 so XLA may run it in place.
        ``1`` — ``(d0, nk, d3, d1, d4, d2)`` emitted by the store, for a
        consumer whose next op wants that layout; no alias (different shape).

    FACTORY-time refusals: mesh platform, missing handler, malformed specs.
    TRACE-time refusals: dtype, rank, extents — trace-time facts, and the
    two-phase contract (``docs/dev/ffi_gate_contract.md`` §1.5) says a
    resolver that claimed to check them earlier would be lying.
    """
    if out_layout not in (0, 1):
        raise ValueError(
            f"out_layout must be 0 (shape(X), aliasable in place) or 1 "
            f"((d0,nk,d3,d1,d4,d2), the consumer permutation); got "
            f"{out_layout!r}.")
    require_conv_kminor(mesh)
    nkx, nky, nkz = (int(v) for v in kgrid)
    nk = nkx * nky * nkz
    xax = tuple(x_spec)
    if len(xax) != 6:
        raise ValueError(
            f"X spec must be rank 6 (d0,d1,d2,d3,d4,nk); got {x_spec}.")
    if xax[5] is not None:
        raise ValueError(
            f"X spec {x_spec} shards the k axis.  The transform is "
            f"device-local: the minor k axis must be REPLICATED (None) — the "
            f"same contract as validate_flat_spec enforces for the k-strided "
            f"member and the XLA helpers.")
    kax = tuple(k_spec)
    if len(kax) != 3 or kax[2] is not None:
        raise ValueError(
            f"K spec must be rank 3 (d1,d2,nk) with k replicated; got "
            f"{k_spec}.")
    if (kax[0], kax[1]) != (xax[1], xax[2]):
        raise ValueError(
            f"K spec {k_spec} places (d1,d2) on {(kax[0], kax[1])} but X spec "
            f"{x_spec} places them on {(xax[1], xax[2])}.  The handler "
            f"multiplies rank-local tiles and implements no reshard; make the "
            f"two agree at the call site.")
    o_spec = conv_kminor_out_spec(x_spec, out_layout)

    attrs = dict(nkx=np.int64(nkx), nky=np.int64(nky), nkz=np.int64(nkz),
                 scale=np.float64(conv_kminor_scale(norm, nk, mult)),
                 out_layout=np.int64(out_layout))

    def _local(x_local, k_local):
        if x_local.ndim != 6 or k_local.ndim != 3:
            raise ValueError(
                f"conv_kminor expects local X (d0,d1,d2,d3,d4,nk) and K "
                f"(d1,d2,nk); got {x_local.shape} / {k_local.shape}.")
        if (x_local.shape[1] != k_local.shape[0]
                or x_local.shape[2] != k_local.shape[1]
                or x_local.shape[5] != k_local.shape[2]):
            raise ValueError(
                f"conv_kminor X/K shard shapes disagree: {x_local.shape} vs "
                f"{k_local.shape} (need X[1]==K[0], X[2]==K[1], X[5]==K[2]).")
        out_t = jax.ShapeDtypeStruct(
            conv_kminor_out_shape(x_local.shape, out_layout), x_local.dtype)
        # The alias is legal ONLY at out_layout=0, where the result has the
        # operand's shape.  At out_layout=1 the store is a permutation and the
        # buffers genuinely differ; claiming an alias there would be a lie XLA
        # would decline anyway.
        kw = {"input_output_aliases": {0: 0}} if out_layout == 0 else {}
        return jax.ffi.ffi_call(CONV_KMINOR_TARGET, out_t, **kw)(
            x_local, k_local, **attrs)

    from common.shard_map import shard_map     # see the import-cycle note
    _sm = shard_map(_local, mesh=mesh,
                    in_specs=(x_spec, k_spec), out_specs=o_spec,
                    check_vma=False)

    def _conv_kminor(X, K):
        if X.dtype != jnp.complex128 or K.dtype != jnp.complex128:
            raise TypeError(
                f"conv_kminor is complex128 ONLY and does not up-cast; got "
                f"X={X.dtype}, K={K.dtype}.  A c64 caller (the fp32-GMRES "
                f"ladder arm casts its payload in "
                f"bse_feast._build_gmres_data_fp32) must refuse here rather "
                f"than silently change the arithmetic it is measuring.")
        if X.ndim != 6 or K.ndim != 3:
            raise ValueError(
                f"conv_kminor expects X rank 6 and K rank 3; got {X.shape} / "
                f"{K.shape}.")
        if int(X.shape[5]) != nk or int(K.shape[2]) != nk:
            raise ValueError(
                f"conv_kminor minor extents {X.shape[5]}/{K.shape[2]} != "
                f"nkx*nky*nkz = {nk}.")
        return _sm(X, K)

    return _conv_kminor
