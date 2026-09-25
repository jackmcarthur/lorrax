"""Batched flat-k 3-D FFT (the ``LORRAX_FFT_FFI`` service) and the k-convolution
router (decisions.md 2026-09-24).

The Python half of the flat-k FFT handlers:

    cpu   liblorrax_ffi_host.so   the FFTW3 ABI (``fftw_plan_many_dft``, the
                                  advanced-layout planner) in
                                  (``src/ffi/cpp/mklfft/fft_flat_k_ffi.cc``) —
                                  a genuine O(N log N) FFT at any k-count.
                                  The directory is still named ``mklfft`` for the
                                  DFTI implementation it USED to hold; the
                                  DFTI calls were deleted 2026-08-05 and the
                                  library is now bound by ``dlsym`` over a
                                  candidate ladder (``LORRAX_FFTW3_SO``).
    CUDA  liblorrax_ffi.so        the k-convolution router's nvidia-mathdx
                                  k-leading transform (mode 3 of
                                  ``src/ffi/cpp/cufft/kconv_mathdx_cuda_ffi.cc``)
                                  since 2026-09-24; the cuFFT advanced-layout
                                  handler it replaced was measured 1.8-7.4x
                                  slower at the production tiles and deleted.

Contract: ``docs/architecture/services.md`` (``ffi.fft``); the k-convolution
router: ``docs/architecture/ffi_layout.md``.

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
THE k-CONVOLUTION ROUTER — one front door per operation, one backend per platform
================================================================================
Every k-axis convolution and k-axis transform the physics needs is asked for
through a factory here (or its ``common.fft_helpers`` alias), and the factory
chooses the backend from the MESH PLATFORM only — never from an environment
variable (decisions.md 2026-09-24, QUALITY #8):

    CUDA   nvidia-mathdx: cuFFTDx thread FFTs inside one fused shared-memory
           pass per k-row, NVRTC-built per (mode, k-grid) and disk-cached
           (``cpp/cufft/kconv_mathdx_cuda_ffi.cc``).  The ONLY NVIDIA backend.
    cpu    the plan route: the FFTW3-ABI flat-k handler (and, for the Σ
           convolution, the fused FFTW gw_conv handler).
    other  refusal by name.

    door                      layout         operation
    ------------------------  -------------  ----------------------------------------
    make_fused_conv_kpair     3-D leading    ISDF CCT/ZCT post-pair convolution
    make_fused_conv_kparent   parent tables  the same with the typed parent load
    make_fused_conv_kplane    route-G planes the same read from the D-plane FFT output,
                                             Bloch phase and L/R split applied on load
    make_kconv_klead          flat leading   Σ / COHSEX  fftn(ifftn(T)·ifftn(W))
    make_kconv_klead_unfold   parent Green   the Σ one read from the raw-parent G with
                                             the typed unfold and spin action on load,
                                             stored at the caller's k rows only
    make_kconv_lorentz_unfold parent Green   the four-current Σ: the same load, then
                                             the γ_i Ĝ γ_j† · V_ij block sum in R space,
                                             stored at the caller's k rows only
    make_kfft_klead_unfold    wedge interaction  make_kconv_klead's prep (ifftn into R
                                             space) read from the q wedge with the
                                             typed unfold on load
    make_kconv_kminor         trailing       BSE rung    fftn(ifftn(X)·K_R)
    make_kfft_klead / _local  flat leading   one transform
    make_kfft_kminor / _local trailing       one transform

Pick the door whose k position matches the tile you already hold; a caller does
not transpose to reach another.  A k-grid axis above ``KCONV_AXIS_MAX`` (40,
the fp64 cuFFTDx thread-FFT limit) or a row that does not fit shared memory is
refused by name on CUDA.

The plain flat-k transform is the same router: ``common.fft_helpers.
make_flat_k_fft`` calls :func:`make_kfft_klead` (mathdx mode 3 on CUDA,
measured 1.8-7.4x faster than the cuFFT plan it replaced; the FFTW3-ABI host
handler on cpu, whose sharded door :func:`make_flat_k_fft_ffi` is cpu-only).
"""

from __future__ import annotations

import math
from typing import Callable, NamedTuple

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
    "FLAT_K_TARGET", "GW_CONV_TARGET", "GATE",
    "fft_ffi_enabled", "fft_ffi_mode",
    "require_fft_ffi", "make_flat_k_fft_ffi", "make_local_flat_k_fft_ffi",
    "ffi_fft_scale", "validate_flat_spec",
    # The k-convolution router (decisions.md 2026-09-24): CUDA -> nvidia-mathdx,
    # cpu -> the plan route.
    "KCONV_PAIR_TARGET", "KCONV_PARENT_TARGET", "KCONV_KLEAD_TARGET",
    "KFFT_KLEAD_TARGET", "KCONV_KMINOR_TARGET", "KFFT_KMINOR_TARGET",
    "KCONV_TARGETS", "KCONV_AXIS_MAX",
    "kconv_backend", "require_kconv", "mathdx_root", "cubin_cache_dir", "conv_kpair_scale",
    "make_fused_conv_kpair", "make_fused_conv_kparent", "make_fused_conv_kplane",
    "KCONV_PLANE_TARGET",
    "KConvStored", "make_kconv_klead", "make_kconv_klead_unfold", "KCONV_KLEAD_UNFOLD_TARGET",
    "make_kconv_lorentz_unfold", "KCONV_KLEAD_LORENTZ_TARGET",
    "make_kfft_klead_unfold", "KFFT_KLEAD_UNFOLD_TARGET",
    "make_kconv_kminor", "kconv_kminor_out_shape",
    "make_kfft_klead", "make_kfft_kminor",
    "make_local_kfft_klead", "make_local_kfft_kminor", "make_local_kconv_kminor",
    "PLANE_FFT_GATHER_TARGET", "plane_fft_split", "plane_resident_bytes", "make_plane_fft_gather",
]

FLAT_K_TARGET = "lorrax_mklfft_flat_k"
#: The FFTW3-ABI fused Σ convolution, the cpu leg of :func:`make_kconv_klead`.
#: HOST ONLY since 2026-09-24: its cuFFT strided CUDA twin was deleted when
#: the router moved CUDA onto nvidia-mathdx.  The name is historical, coined
#: by the CPU prototype.
GW_CONV_TARGET = "lorrax_mklfft_gw_conv"
#: The NVIDIA k-convolution family on nvidia-mathdx (the router's CUDA leg).
KCONV_PAIR_TARGET = "lorrax_mathdx_kconv_pair"
KCONV_PARENT_TARGET = "lorrax_mathdx_kconv_parent"
KCONV_PLANE_TARGET = "lorrax_mathdx_kconv_plane"
KCONV_KLEAD_TARGET = "lorrax_mathdx_kconv_klead"
KCONV_KLEAD_UNFOLD_TARGET = "lorrax_mathdx_kconv_klead_unfold_rows"
KCONV_KLEAD_LORENTZ_TARGET = "lorrax_mathdx_kconv_klead_lorentz_rows"
KFFT_KLEAD_UNFOLD_TARGET = "lorrax_mathdx_kfft_klead_unfold"
KFFT_KLEAD_TARGET = "lorrax_mathdx_kfft_klead"
KCONV_KMINOR_TARGET = "lorrax_mathdx_kconv_kminor"
KFFT_KMINOR_TARGET = "lorrax_mathdx_kfft_kminor"
#: Mode 10, the route-G plane FFT with gather-on-load (:func:`make_plane_fft_gather`).
PLANE_FFT_GATHER_TARGET = "lorrax_mathdx_plane_fft_gather"
#: Every mathdx target; ``require_kconv`` checks them all at startup.
KCONV_TARGETS = (KCONV_PAIR_TARGET, KCONV_PARENT_TARGET, KCONV_PLANE_TARGET, KCONV_KLEAD_TARGET,
                 KCONV_KLEAD_UNFOLD_TARGET, KCONV_KLEAD_LORENTZ_TARGET,
                 KFFT_KLEAD_TARGET, KFFT_KLEAD_UNFOLD_TARGET, KCONV_KMINOR_TARGET,
                 KFFT_KMINOR_TARGET, PLANE_FFT_GATHER_TARGET)

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
    # cpu ONLY since 2026-09-24: on CUDA the flat-k transform is the
    # k-convolution router's nvidia-mathdx k-leading mode (make_kfft_klead),
    # measured 1.8-7.4x faster than the cuFFT advanced-layout plan it replaced
    # (runs/runtime/kconv_stage2_20260924/bench_flatk_v2.log).
    platforms=("cpu",),
    silent_platform_demote=(
        "on CUDA the flat-k transform is the k-convolution router's "
        "nvidia-mathdx family, checked at startup by require_kconv"),
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
        "on cpu, the nvidia-mathdx k-convolution router on CUDA).  Unset "
        "LORRAX_FFT_FFI, or recover the XLA arm from git history for a "
        "debugging build."),
    # NOTE the platform names below are ABIs, not products.  The host handler
    # calls the FFTW3 ABI (`fftw_plan_many_dft` ×4 in
    # cpp/mklfft/fft_flat_k_ffi.cc; zero `DftiCreateDescriptor` since
    # 2026-08-05) and binds it by dlsym against whatever the process links —
    # cray-fftw, a system FFTW3, or MKL's FFTW3 wrappers via `libmkl_rt.so`.
    # These strings said "MKL FFT (DFTI API)" for the five days after the
    # DFTI code was deleted, so every CPU startup block named an engine the
    # translation unit no longer contained.  Name the ABI; let
    # LORRAX_DEBUG_PRINT name the library.
    label={"cpu": "FFTW3-ABI host"},
    resolved_msg={
        "cpu": ("[fft_ffi] flat-k 3-D FFTs -> FFTW3-ABI host FFI handler "
                "({target}): O(N log N) FFT reading the dot-layout tile "
                "in place via advanced-layout plans — no XLA layout "
                "transposes.  WHICH library answers is resolved at run "
                "time by dlsym over the candidate ladder (see "
                "LORRAX_FFTW3_SO and docs/architecture/ffi_layout.md §3), "
                "and is NOT stated by this line."),
    },
    refuse_platform_msg=(
        "LORRAX_FFT_FFI: the required FFI flat-k FFT backend cannot serve "
        "this mesh — its devices are '{platform}'; this gate serves cpu (the "
        "FFTW3 ABI) only, and CUDA meshes take the k-convolution router "
        "(nvidia-mathdx)."),
    refuse_probe_msg=(
        "The required {label} backend is unavailable: FFI target "
        "'{target}' is unusable on platform '{platform}': {reason}  The "
        "FFI layer is REQUIRED (docs/architecture/decisions.md, "
        "2026-08-01); build/locate the library per "
        "docs/environment/overview.md (host: build_host.sh -> "
        "liblorrax_ffi_host.so, selected by LORRAX_FFI_HOST_SO)."),
)

def fft_ffi_mode() -> str:
    """``"on"`` | ``"off"`` — the raw ``LORRAX_FFT_FFI`` grammar."""
    return GATE.mode()


def fft_ffi_enabled() -> bool:
    """True when ``make_flat_k_*`` should return the FFI variant.

    Read at helper-FACTORY time; kernel caches must key on it
    (``gw.ppm_tau_kernel``).  Backend-init-free (gate contract tier 1)."""
    return GATE.enabled()


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


def make_local_flat_k_fft_ffi(
    kgrid: tuple[int, int, int],
    *,
    kind: str,
    norm: str | None,
) -> Callable:
    """Return the flat-k FFI call for use inside an existing ``shard_map``.

    This is the local kernel owned by :func:`make_flat_k_fft_ffi`, exposed so
    an already manually sharded caller can bound a trailing-axis workspace
    without nesting another ``shard_map``.  Platform/target validation remains
    the outer factory's responsibility.
    """
    if kind not in ('ifftn', 'fftn'):
        raise ValueError(f"kind must be 'ifftn' or 'fftn', got {kind!r}")
    nkx, nky, nkz = (int(v) for v in kgrid)
    nk = nkx * nky * nkz
    scale = ffi_fft_scale(kind, norm, nk)
    attrs = dict(nkx=np.int64(nkx), nky=np.int64(nky), nkz=np.int64(nkz),
                 forward=np.int64(0 if kind == 'ifftn' else 1),
                 scale=np.float64(scale))

    def _local(x_local):
        if x_local.dtype != jnp.complex128:
            raise TypeError(
                "FFI flat-k backend supports complex128 only, got "
                f"{x_local.dtype}.")
        if int(x_local.shape[0]) != nk:
            raise ValueError(
                f"flat-k local input leading extent {x_local.shape[0]} != "
                f"nkx*nky*nkz = {nk}.")
        out_t = jax.ShapeDtypeStruct(x_local.shape, x_local.dtype)
        return jax.ffi.ffi_call(
            FLAT_K_TARGET, out_t,
            input_output_aliases={0: 0},
        )(x_local, **attrs)

    return _local


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
    over the local shard (the FFTW3-ABI host handler; this door is cpu-only),
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
    _local = make_local_flat_k_fft_ffi(kgrid, kind=kind, norm=norm)

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
# The cpu leg of the k-leading convolution: the FFTW3-ABI gw_conv host handler
# ===========================================================================
def _host_gw_conv_local(kgrid, norm: str | None, mult: float) -> Callable:
    """Rank-local ``fn(G, W) -> sigma`` on the host gw_conv handler (the cpu plan route).

    ``sigma = fftn(ifftn(G) * ifftn(W)[:, None, :, None, :] * mult)`` with all
    three FFTW advanced-layout transforms and the broadcast multiply in one
    call, chunked so the R-space G tile never materialises.  ``G``/``sigma``
    ``(nk, a, mx, b, my)``, ``W`` ``(nk, mx, my)``.  Only
    :func:`make_kconv_klead` builds it (its cpu leg); CUDA meshes take the
    nvidia-mathdx family instead.
    """
    nkx, nky, nkz = (int(v) for v in kgrid)
    nk = nkx * nky * nkz
    attrs = dict(nkx=np.int64(nkx), nky=np.int64(nky), nkz=np.int64(nkz),
                 scale_i=np.float64(ffi_fft_scale('ifftn', norm, nk)),
                 scale_f=np.float64(ffi_fft_scale('fftn', norm, nk) * float(mult)))

    def _local(g_local, w_local):
        out_t = jax.ShapeDtypeStruct(g_local.shape, g_local.dtype)
        return jax.ffi.ffi_call(
            GW_CONV_TARGET, out_t,
            input_output_aliases={0: 0},  # sigma_k in G_k's buffer when dead
        )(g_local, w_local, **attrs)

    return _local


# ===========================================================================
# THE k-CONVOLUTION ROUTER — the ISDF pair convolution (decisions.md 2026-09-24)
# ===========================================================================
# Physics code asks for a pair convolution; the router answers by PLATFORM
# only, never by an environment variable:
#
#     CUDA  -> the nvidia-mathdx family (cpp/cufft/kconv_mathdx_cuda_ffi.cc):
#              cuFFTDx transforms, one fused pass, NVRTC-built per k-grid.
#     cpu   -> the MKL flat-k plan route (``lorrax_mklfft_flat_k``) composed
#              with XLA elementwise spin contraction.
#     other -> refusal.
#
# Both backends return the SAME callable contract, so a consumer never
# branches on the backend.  There is no plan route on NVIDIA.

#: cuFFTDx fp64 thread-FFT limit: every k-grid axis must be at most this.
KCONV_AXIS_MAX = 40


def conv_kpair_scale(norm: str | None, nk: int, mult: float = 1.0) -> float:
    """Fold two inverse norms and one forward norm into one scalar."""
    si = ffi_fft_scale("ifftn", norm, nk)
    return si * si * ffi_fft_scale("fftn", norm, nk) * float(mult)


def _conv_kpair_phase_codes(phase, ns: int, label: str) -> np.ndarray:
    """Encode exact monomial phases as 0:+1, 1:+i, 2:-1, 3:-i."""
    values = np.asarray(phase, dtype=np.complex128).reshape(-1)
    if values.size != ns:
        raise ValueError(f"k-conv {label} phase has {values.size} entries; ns={ns}")
    quadrants = np.asarray([1, 1j, -1, -1j], dtype=np.complex128)
    codes = np.empty(ns, dtype=np.int64)
    for i, value in enumerate(values):
        hits = np.flatnonzero(np.abs(quadrants - value) <= 1e-14)
        if hits.size != 1:
            raise ValueError(
                f"k-conv {label} phase[{i}]={value!r} is not in {{+1,+i,-1,-i}}")
        codes[i] = int(hits[0])
    return codes


def _check_perm(perm, ns: int, label: str) -> np.ndarray:
    p = np.asarray(perm, dtype=np.int64).reshape(-1)
    if p.size != ns or sorted(int(v) for v in p) != list(range(ns)):
        raise ValueError(f"k-conv {label} perm {p.tolist()} is not a permutation of range({ns})")
    return p


def mathdx_root() -> str:
    """The installed nvidia-mathdx wheel's ``nvidia/mathdx`` directory, or a GATE refusal.

    Found from the Python package spec, never from an environment variable;
    NVRTC includes ``include/`` and ``external/cutlass/include`` beneath it.
    """
    import importlib.util
    import os
    try:
        spec = importlib.util.find_spec("nvidia.mathdx")
    except ModuleNotFoundError:
        spec = None
    roots = list(spec.submodule_search_locations) if spec is not None else []
    for root in roots:
        if os.path.isfile(os.path.join(root, "include", "cufftdx.hpp")):
            return root
    raise RuntimeError(
        "GATE mathdx-headers: got no importable nvidia.mathdx with "
        "include/cufftdx.hpp; want the nvidia-mathdx wheel, the only supported "
        "k-convolution backend on NVIDIA GPUs (decisions.md 2026-09-24); why: "
        "the fused k-convolution kernels are compiled at run time from its "
        "cuFFTDx headers; fix: pip install nvidia-mathdx.")


def kconv_backend(mesh: Mesh) -> str:
    """``'mathdx'`` on a CUDA mesh, ``'plan'`` on a cpu mesh; any other platform refuses."""
    from ffi.gate import mesh_ffi_platform
    plat = mesh_ffi_platform(mesh)
    if plat == "CUDA":
        return "mathdx"
    if plat == "cpu":
        return "plan"
    raise RuntimeError(
        f"GATE kconv-platform: got a {plat!r} mesh; want CUDA (nvidia-mathdx) "
        "or cpu (MKL flat-k plans); why: the k-convolution router has no backend "
        "for this platform; fix: run on a supported platform.")


def require_kconv(mesh: Mesh, *, announce: bool = True) -> str:
    """Startup check of the router's backend on this mesh; returns it or refuses.

    CUDA: the nvidia-mathdx wheel and both family targets; cpu: the flat-k
    plan target (already required by ``LORRAX_FFT_FFI``).
    """
    from ffi.gate import announce_once
    backend = kconv_backend(mesh)
    if backend == "mathdx":
        root = mathdx_root()
        for target in KCONV_TARGETS:
            _require_target(target, "CUDA")
        _probe_kconv_compile(mesh)
        announce_once(("kconv", "backend", backend),
                      f"[kconv] k-convolution router: CUDA -> nvidia-mathdx ({root}); "
                      f"cubin cache {_cubin_cache_summary()}",
                      scope="rank0", emit=announce)
    else:
        _require_plan_route()
        announce_once(("kconv", "backend", backend),
                      "[kconv] k-convolution router: cpu -> MKL flat-k plan route",
                      scope="rank0", emit=announce)
    return backend


def _probe_kconv_compile(mesh: Mesh) -> None:
    """Compile and run one tiny mathdx kernel (mode 3, k-grid 2x1x1) on this
    process's first mesh device (else its first local device), so a device the installed cuFFTDx cannot
    compile for refuses at startup, naming its compute capability, rather than
    at the first k-convolution.  The cubin is disk-cached like every other."""
    local = [d for d in mesh.devices.flat if d.process_index == jax.process_index()]
    dev = local[0] if local else jax.local_devices()[0]
    try:
        with jax.default_device(dev):
            jax.block_until_ready(jax.jit(lambda: _rows_kfft_call(
                KFFT_KLEAD_TARGET, jnp.zeros((2, 1), jnp.complex128), (2, 1, 1),
                forward=True, scale=1.0))())
    except Exception as e:                                          # noqa: BLE001
        from importlib import metadata
        try:
            wheel = metadata.version("nvidia-mathdx")
        except metadata.PackageNotFoundError:
            wheel = "?"
        cc = getattr(dev, "compute_capability", "?")
        raise RuntimeError(
            f"GATE mathdx-probe: got a probe compile failure on {dev.device_kind} (compute "
            f"capability {cc}) with nvidia-mathdx {wheel}; want every mathdx kernel to compile "
            "for this device; why: cuFFTDx defines a fixed list of SM targets and the kernels "
            "are built by NVRTC for the device's own; fix: a nvidia-mathdx wheel that supports "
            f"this architecture. Cause: {e}") from e


def _cubin_cache_summary() -> str:
    """``<dir>: N images, X MB`` for the startup line (one flat directory, no walk)."""
    import os
    d = cubin_cache_dir()
    try:
        sizes = [e.stat().st_size for e in os.scandir(d)
                 if e.is_file() and e.name.endswith(".cubin")]
    except FileNotFoundError:
        sizes = []
    return f"{d}: {len(sizes)} images, {sum(sizes) / 1e6:.1f} MB"


def _require_target(target: str, platform: str) -> None:
    from ffi.common import ffi_loader
    ok, why = ffi_loader.probe_target(target, platform)
    if not ok:
        raise RuntimeError(
            f"GATE kconv-target: got a liblorrax_ffi without {target} on "
            f"{platform} ({why}); want the handler this router selects; fix: "
            "rebuild the native library (src/ffi/cpp/build.sh) and point "
            "LORRAX_FFI_SO at it.")


def _mathdx_attrs(kgrid, ns, perm_l, phase_l, perm_r, phase_r, scale) -> dict:
    kg = _check_kgrid(kgrid, "mathdx")
    return dict(nkx=np.int64(kg[0]), nky=np.int64(kg[1]), nkz=np.int64(kg[2]),
                scale=np.float64(scale),
                perm_l=_check_perm(perm_l, ns, "left"),
                phase_l=_conv_kpair_phase_codes(phase_l, ns, "left"),
                perm_r=_check_perm(perm_r, ns, "right"),
                phase_r=_conv_kpair_phase_codes(phase_r, ns, "right"),
                **_mathdx_common())


def _check_kgrid(kgrid, backend: str) -> tuple[int, int, int]:
    """Three positive k axes; on the ``mathdx`` leg also at most ``KCONV_AXIS_MAX``.

    The cap is the fp64 cuFFTDx thread-FFT limit, a CUDA constraint: the cpu
    plan route (FFTW) has none, so ``backend = "plan"`` does not apply it.
    """
    kg = tuple(int(v) for v in kgrid)
    if len(kg) != 3 or min(kg) < 1:
        raise RuntimeError(
            f"GATE kconv-kgrid: got k-grid {kg}; want three positive axes; fix: "
            "pass the (nkx, nky, nkz) of the run's k-grid.")
    if backend == "mathdx" and max(kg) > KCONV_AXIS_MAX:
        raise RuntimeError(
            f"GATE mathdx-kconv-axis: got k-grid {kg}; want every axis in "
            f"[1,{KCONV_AXIS_MAX}]; why: the fp64 cuFFTDx thread-FFT limit; fix: "
            "a smaller k-grid.")
    return kg


def _mathdx_common() -> dict:
    """The two string attributes every mathdx handler takes: the wheel root and
    the disk cubin cache directory (:func:`cubin_cache_dir`)."""
    return dict(mathdx_root=mathdx_root(), cubin_dir=cubin_cache_dir())


def cubin_cache_dir() -> str:
    """Where the compiled mathdx kernels are kept: ``$SCRATCH/.cache/lorrax/kconv_mathdx``,
    or ``~/.cache/lorrax/kconv_mathdx`` where the site defines no ``SCRATCH``.

    Always on, and separate from the XLA compile cache
    (``common.jax_compile_cache``, one namespace per release): this store is small
    and content-addressed — each image is keyed by the full hash of its source,
    NVRTC options, wheel version and NVRTC version, written by tmp+rename and
    re-hashed on read — so reusing it can never change a result, while
    rebuilding it costs about 6 s per (mode, k-grid) per process.  One
    directory for every world size: an image depends on the device and the
    wheel, not on P.  No knob.
    """
    import os
    root = os.environ.get("SCRATCH") or os.path.expanduser("~")
    return os.path.join(root, ".cache", "lorrax", "kconv_mathdx")


# ---- the cpu leg: the MKL flat-k plan route ---------------------------------

def _plan_kfft(x_flat, kgrid, kind: str):
    """Unnormalised transform of the leading flat-k axis on the cpu plan route.

    TEST-ONLY exception: on the cpu backend with ``LORRAX_KFFT_CPU_TEST_XLA=1``
    (set by ``tests/conftest.py`` for in-process cpu meshes, which have no host
    FFI library on Perlmutter) this announces itself and uses ``jnp.fft``.
    """
    import os
    kg = tuple(int(v) for v in kgrid)
    norm = "forward" if kind == "ifftn" else "backward"          # both unnormalised
    if os.environ.get("LORRAX_KFFT_CPU_TEST_XLA") == "1" and jax.default_backend() == "cpu":
        from ffi.gate import announce_once
        announce_once(("kconv", "cpu-test-xla"),
                      "[kconv] TEST-ONLY: LORRAX_KFFT_CPU_TEST_XLA=1 on cpu -> jnp.fft "
                      "k-axis transforms (never a production path)")
        f = jnp.fft.ifftn if kind == "ifftn" else jnp.fft.fftn
        y = f(x_flat.reshape(kg + tuple(x_flat.shape[1:])), axes=(0, 1, 2), norm=norm)
        return y.reshape(x_flat.shape)
    return make_local_flat_k_fft_ffi(kg, kind=kind, norm=norm)(x_flat)


def _plan_pair_tail(P_l, P_r, kgrid, perm_l, phase_l, perm_r, phase_r, scale):
    """``s·FFT_k Σ_ab phase_l[a]·phase_r[b]·conj(IFFT_k P_l[:,a,…,b])·IFFT_k P_r[:,π_l a,…,π_r b]``.

    ``P_l``/``P_r`` are flat-k open-spin ``(nk, ns, *rows, ns)``; returns ``(nk, *rows)``.
    """
    ns = int(P_l.shape[1])
    I_l = jnp.conj(_plan_kfft(P_l, kgrid, "ifftn"))
    I_r = _plan_kfft(P_r, kgrid, "ifftn")
    Z = 0
    for a in range(ns):
        for b in range(ns):
            w = complex(phase_l[a]) * complex(phase_r[b])
            Z = Z + w * I_l[:, a, ..., b] * I_r[:, int(perm_l[a]), ..., int(perm_r[b])]
    return _plan_kfft(Z, kgrid, "fftn") * scale


def _parent_open_spin(D, tables, right: bool):
    """The typed parent load of every full-k child: ``(nk, ns, mu, nu, ns)``.

    The same map as the mathdx kernel's load: umklapp phases of both
    endpoints, antiunitary conjugation, then ``P = conj(Σ_cd coef·T(D_cd))``.
    """
    irr, sym, left, rightp, L, R, q, trs, coef_l, coef_r = tables
    ns = int(D.shape[1])
    coef = (coef_r if right else coef_l).reshape(-1, ns, ns, ns, ns)   # (k, a, b, c, e)
    lm = jnp.take(left, sym, axis=0)                                    # (k, mu)
    rn = jnp.take(rightp, sym, axis=0)                                  # (k, nu)
    qp = jnp.take(q, irr, axis=0)                                       # (k, 3)
    pl = jnp.exp(2j * jnp.pi * jnp.einsum('ki,kmi->km', qp, jnp.take(L, sym, axis=0)))
    pr = jnp.exp(-2j * jnp.pi * jnp.einsum('ki,kni->kn', qp, jnp.take(R, sym, axis=0)))
    G = jnp.take(D, irr, axis=0)                                        # (k, c, mu, e, nu)
    G = jnp.take_along_axis(G, lm[:, None, :, None, None], axis=2)
    G = jnp.take_along_axis(G, rn[:, None, None, None, :], axis=4)
    V = pl[:, None, :, None, None] * G * pr[:, None, None, None, :]
    V = jnp.where((trs != 0)[:, None, None, None, None], jnp.conj(V), V)
    return jnp.conj(jnp.einsum('kabce,kcmen->kamnb', coef, V))   # spin axes 1 and -1


# ---- the router's pair-convolution factories --------------------------------

def make_fused_conv_kpair(
    mesh: Mesh,
    kgrid: tuple[int, int, int],
    *,
    perm_l,
    phase_l,
    perm_r,
    phase_r,
    norm: str | None = "forward",
    mult: float = 1.0,
) -> Callable:
    """The ISDF CCT/ZCT post-pair convolution ``fn(A, B) -> U``, routed by platform.

    ``A``/``B`` ``(nkx,nky,nkz, ns, col, mu, ns)`` c128, device-local inside
    the caller's ``shard_map``; ``U`` ``(nkx,nky,nkz, col, mu)``::

        U = s·FFT_k Σ_ab phase_l[a]·phase_r[b]·conj(IFFT_k A[…,a,…,b])·IFFT_k B[…,π_l a,…,π_r b]

    CUDA: the nvidia-mathdx family; cpu: the MKL flat-k plan route; the two
    return the same contract (decisions.md 2026-09-24).
    """
    ns = int(np.asarray(perm_l).size)
    nkx, nky, nkz = (int(v) for v in kgrid)
    nk = nkx * nky * nkz
    scale = conv_kpair_scale(norm, nk, mult)
    backend = kconv_backend(mesh)
    if backend == "mathdx":
        _require_target(KCONV_PAIR_TARGET, "CUDA")
        attrs = _mathdx_attrs(kgrid, ns, perm_l, phase_l, perm_r, phase_r, scale)

        def _mathdx(A, B):
            _check_pair_operands(A, B, (nkx, nky, nkz), ns)
            out = jax.ShapeDtypeStruct(A.shape[:3] + A.shape[4:6], A.dtype)
            return jax.ffi.ffi_call(KCONV_PAIR_TARGET, out)(A, B, **attrs)
        return _mathdx

    _require_plan_route()
    return _plan_kpair(kgrid, ns, perm_l, phase_l, perm_r, phase_r, scale)


def make_fused_conv_kparent(mesh, kgrid, ns, trailing_shape, *,
                            perm_l, phase_l, perm_r, phase_r, centroid_major=False) -> Callable:
    """The ISDF parent-load pair convolution ``fn(D_l, D_r, tables) -> U``, routed by platform.

    ``D_l``/``D_r`` ``(n_parent, ns, mu, ns, nu)`` c128 raw-parent projectors
    and the ten typed tables of ``isdf.core._parent_conv_tables_local``;
    ``U`` ``(nk, mu, nu)``.  ``centroid_major`` states the physical layout of
    the D operands (CCT) for the CUDA handler; the logical contract is
    unchanged.  ``trailing_shape`` is the caller's ``(mu, nu)`` tile, kept for
    the seam's signature.  CUDA: nvidia-mathdx; cpu: the MKL plan route.
    """
    del trailing_shape
    ns = int(ns)
    nkx, nky, nkz = (int(v) for v in kgrid)
    nk = nkx * nky * nkz
    scale = conv_kpair_scale("forward", nk, 1.0)
    backend = kconv_backend(mesh)
    if backend == "mathdx":
        _require_target(KCONV_PARENT_TARGET, "CUDA")
        attrs = _mathdx_attrs(kgrid, ns, perm_l, phase_l, perm_r, phase_r, scale)
        attrs["centroid_major"] = np.int64(bool(centroid_major))
        layout = (0, 4, 3, 2, 1) if centroid_major else None

        def _mathdx(D_l, D_r, tables):
            _check_parent_operands(D_l, D_r, ns)
            out = jax.ShapeDtypeStruct((nk, D_l.shape[2], D_l.shape[4]), D_l.dtype)
            return jax.ffi.ffi_call(
                KCONV_PARENT_TARGET, out,
                input_layouts=(layout, layout, *(None for _ in tables)),
            )(D_l, D_r, *tables, **attrs)
        return _mathdx

    _require_plan_route()
    return _plan_kparent(kgrid, ns, perm_l, phase_l, perm_r, phase_r, scale)


def make_fused_conv_kplane(mesh, kgrid, ns, *, perm_l, phase_l, perm_r, phase_r) -> Callable:
    """The route-G pair convolution ``fn(D, F) -> U`` read from the D-plane FFT output.

    ``D`` ``(nk, g, ns, 2c, ns, p)`` c128 is the plane transform exactly as
    ``isdf.zeta_mubatch`` leaves it (``g`` planes of ``p`` points; slots
    ``[0, c)`` of the ``2c`` axis are the L projector, ``[c, 2c)`` the R one),
    ``F`` ``(nk, g, p)`` its Bloch phase; ``U`` ``(nk, c, g·p)``::

        P^X_{k,ab}(m, (g,p)) = conj(F[k,g,p] · D[k,g,a,X+m,b,p])      X = 0 | c
        U = s·FFT_k Σ_ab phase_l[a]·phase_r[b]·conj(IFFT_k P^L_ab)·IFFT_k P^R_{π_l a, π_r b}

    — :func:`make_fused_conv_kparent` on the identity plan, without the
    transposed, phased and split copy of ``D`` that its operand layout needs.
    CUDA: nvidia-mathdx mode 6 (the phase and the split are applied on load);
    cpu: the plan route on the same composition.  ``s`` is the forward-norm
    pair scale of the parent door.
    """
    ns = int(ns)
    nkx, nky, nkz = (int(v) for v in kgrid)
    nk = nkx * nky * nkz
    scale = conv_kpair_scale("forward", nk, 1.0)
    if kconv_backend(mesh) == "mathdx":
        _require_target(KCONV_PLANE_TARGET, "CUDA")
        attrs = _mathdx_attrs(kgrid, ns, perm_l, phase_l, perm_r, phase_r, scale)

        def _mathdx(D, F):
            c = _check_plane_operands(D, F, nk, ns)
            out = jax.ShapeDtypeStruct((nk, c, D.shape[1] * D.shape[5]), D.dtype)
            return jax.ffi.ffi_call(KCONV_PLANE_TARGET, out)(D, F, **attrs)
        return _mathdx

    _require_plan_route()
    pl, pr = _check_perm(perm_l, ns, "left"), _check_perm(perm_r, ns, "right")
    phl = np.asarray(phase_l, np.complex128).reshape(-1)
    phr = np.asarray(phase_r, np.complex128).reshape(-1)

    def _plan(D, F):
        c = _check_plane_operands(D, F, nk, ns)
        X = jnp.moveaxis(D * F[:, :, None, None, None, :], 1, 4)    # (k, a, 2c, b, g, p)
        X = jnp.conj(jnp.moveaxis(X.reshape(nk, ns, 2 * c, ns, -1), 3, 4))
        return _plan_pair_tail(X[:, :, :c], X[:, :, c:], kgrid, pl, phl, pr, phr, scale)
    return _plan


def _check_plane_operands(D, F, nk: int, ns: int) -> int:
    """Shape/dtype contract of :func:`make_fused_conv_kplane`; returns ``c``."""
    if (D.ndim != 6 or F.ndim != 3 or D.dtype != jnp.complex128 or F.dtype != jnp.complex128
            or int(D.shape[0]) != nk or int(D.shape[2]) != ns or int(D.shape[4]) != ns
            or int(D.shape[3]) % 2 or tuple(F.shape) != (nk, D.shape[1], D.shape[5])):
        raise ValueError(
            f"k-conv plane expects c128 D (nk={nk}, g, ns={ns}, 2c, ns, p) and F (nk, g, p); "
            f"got {D.shape} {D.dtype} / {F.shape} {F.dtype}")
    return int(D.shape[3]) // 2


def _plan_kpair(kgrid, ns, perm_l, phase_l, perm_r, phase_r, scale) -> Callable:
    """The plan-route ``fn(A, B) -> U`` of :func:`make_fused_conv_kpair` (the cpu leg)."""
    kg = tuple(int(v) for v in kgrid)
    nk = kg[0] * kg[1] * kg[2]
    pl, pr = _check_perm(perm_l, ns, "left"), _check_perm(perm_r, ns, "right")
    phl = np.asarray(phase_l, np.complex128).reshape(-1)
    phr = np.asarray(phase_r, np.complex128).reshape(-1)

    def _plan(A, B):
        _check_pair_operands(A, B, kg, ns)
        flat = lambda X: X.reshape((nk,) + tuple(X.shape[3:]))
        U = _plan_pair_tail(flat(A), flat(B), kg, pl, phl, pr, phr, scale)
        return U.reshape(A.shape[:3] + A.shape[4:6])
    return _plan


def _plan_kparent(kgrid, ns, perm_l, phase_l, perm_r, phase_r, scale) -> Callable:
    """The plan-route ``fn(D_l, D_r, tables) -> U`` of :func:`make_fused_conv_kparent`."""
    pl, pr = _check_perm(perm_l, ns, "left"), _check_perm(perm_r, ns, "right")
    phl = np.asarray(phase_l, np.complex128).reshape(-1)
    phr = np.asarray(phase_r, np.complex128).reshape(-1)

    def _plan(D_l, D_r, tables):
        _check_parent_operands(D_l, D_r, ns)
        return _plan_pair_tail(_parent_open_spin(D_l, tables, False),
                               _parent_open_spin(D_r, tables, True),
                               kgrid, pl, phl, pr, phr, scale)
    return _plan


def _require_plan_route() -> None:
    """The cpu leg needs the host flat-k handler (or the announced test-only arm)."""
    import os
    if os.environ.get("LORRAX_KFFT_CPU_TEST_XLA") == "1":
        return
    if not fft_ffi_enabled():
        raise RuntimeError(GATE.off_refuse_msg)
    _require_target(FLAT_K_TARGET, "cpu")


def _check_pair_operands(A, B, kg, ns) -> None:
    if A.dtype != jnp.complex128 or B.dtype != jnp.complex128:
        raise TypeError(f"k-conv pair is complex128 only; got {A.dtype}/{B.dtype}")
    if A.ndim != 7 or A.shape != B.shape or tuple(int(v) for v in A.shape[:3]) != kg \
            or int(A.shape[3]) != ns or int(A.shape[6]) != ns:
        raise ValueError(
            f"k-conv pair expects equal (nkx,nky,nkz,ns,col,mu,ns) operands with "
            f"k-grid {kg}, ns={ns}; got {A.shape}/{B.shape}")


def _check_parent_operands(D_l, D_r, ns) -> None:
    if (D_l.ndim != 5 or D_l.shape != D_r.shape or D_l.shape[1] != ns or D_l.shape[3] != ns
            or D_l.dtype != jnp.complex128 or D_r.dtype != jnp.complex128):
        raise ValueError("k-conv parent requires matching c128 (parent,ns,mu,ns,nu) operands")


# ===========================================================================
# THE k-CONVOLUTION ROUTER — stored-kernel convolutions and k-axis transforms
# ===========================================================================
# Same routing as the pair family above: CUDA -> nvidia-mathdx (modes 2-5 of
# cpp/cufft/kconv_mathdx_cuda_ffi.cc), cpu -> the plan route, anything else ->
# refusal.  The k axis is either LEADING (the Σ/COHSEX dot layout, flat k first)
# or MINOR (the BSE ring layout, the three k axes last); a caller asks for the
# door that matches the tile it holds, and never transposes to reach another.
#
#     make_kconv_klead   Σ, COHSEX   U = mult·fftn(ifftn(T)·ifftn(W)[:,None,:,None,:])
#     make_kconv_kminor  BSE rung    U = mult·fftn_k(ifftn_k(X)·K_R)   (K_R already R space)
#     make_kfft_klead    flat-k      Y = fftn|ifftn over the leading k axis
#     make_kfft_kminor   BSE         Y = fftn|ifftn over the three trailing k axes
#
# The *_local variants are the same callables for code already inside a
# shard_map (which cannot nest); the plain ones wrap their own.

class KConvStored(NamedTuple):
    """A stored-kernel k-leading convolution, split at its one W-only seam.

    ``prep(W) -> W_prep`` is everything that depends on W alone, paid once per
    W; ``apply(T, W_prep) -> U`` is the rest, paid per T.  ``W_prep`` is the
    backend's own form (R space on CUDA, k space on the cpu handler, which
    transforms W itself): pass it only to the ``apply`` of the same pair.
    """
    prep: Callable
    apply: Callable


def _cpu_test_arm() -> bool:
    """The announced TEST-ONLY jnp arm of the cpu leg (see :func:`_plan_kfft`)."""
    import os
    return (os.environ.get("LORRAX_KFFT_CPU_TEST_XLA") == "1"
            and jax.default_backend() == "cpu")


def _rows_kfft_call(target, x2, kg, *, forward: bool, scale: float):
    """One mathdx transform call on a 2-D (nk, rows) or (rows, nk) tile."""
    attrs = dict(nkx=np.int64(kg[0]), nky=np.int64(kg[1]), nkz=np.int64(kg[2]),
                 scale=np.float64(scale), forward=np.int64(1 if forward else 0),
                 **_mathdx_common())
    return jax.ffi.ffi_call(target, jax.ShapeDtypeStruct(x2.shape, x2.dtype),
                            input_output_aliases={0: 0})(x2, **attrs)


def _check_complex(*xs):
    """Every operand complex128, or every operand complex64 (the fp32-GMRES BSE
    arm; mathdx modes 2-5 compile a single-precision image for it).  Never cast."""
    dts = {jnp.dtype(x.dtype) for x in xs}
    if len(dts) != 1 or dts.pop() not in (jnp.dtype(jnp.complex128), jnp.dtype(jnp.complex64)):
        raise TypeError(f"the k-convolution router takes all-complex128 or all-complex64 "
                        f"operands; got {[str(x.dtype) for x in xs]} (it never casts)")


def make_local_kfft_klead(mesh: Mesh, kgrid, *, kind: str, norm: str | None) -> Callable:
    """Rank-local ``fn(X) -> Y`` over the LEADING flat-k axis of ``X (nk, *trail)``."""
    if kind not in ("ifftn", "fftn"):
        raise ValueError(f"kind must be 'ifftn' or 'fftn', got {kind!r}")
    kg = _check_kgrid(kgrid, kconv_backend(mesh))
    nk = kg[0] * kg[1] * kg[2]
    scale = ffi_fft_scale(kind, norm, nk)
    if kconv_backend(mesh) == "mathdx":
        _require_target(KFFT_KLEAD_TARGET, "CUDA")

        def _mathdx(x):
            _check_complex(x)
            y = _rows_kfft_call(KFFT_KLEAD_TARGET, x.reshape(nk, -1), kg,
                                forward=kind == "fftn", scale=scale)
            return y.reshape(x.shape)
        return _mathdx
    _require_plan_route()
    if not _cpu_test_arm():
        return make_local_flat_k_fft_ffi(kg, kind=kind, norm=norm)   # the host plan handler

    def _plan(x):
        _check_complex(x)
        return _plan_kfft(x, kg, kind) * scale
    return _plan


def make_local_kfft_kminor(mesh: Mesh, kgrid, *, kind: str, norm: str | None) -> Callable:
    """Rank-local ``fn(X) -> Y`` over the three TRAILING k axes of ``X (..., nkx, nky, nkz)``."""
    if kind not in ("ifftn", "fftn"):
        raise ValueError(f"kind must be 'ifftn' or 'fftn', got {kind!r}")
    kg = _check_kgrid(kgrid, kconv_backend(mesh))
    nk = kg[0] * kg[1] * kg[2]
    scale = ffi_fft_scale(kind, norm, nk)
    mathdx = kconv_backend(mesh) == "mathdx"
    if mathdx:
        _require_target(KFFT_KMINOR_TARGET, "CUDA")
    else:
        _require_plan_route()

    def _kfft(x):
        _check_complex(x)
        if tuple(int(v) for v in x.shape[-3:]) != kg:
            raise ValueError(f"k-minor transform expects trailing k axes {kg}; got {x.shape}")
        if mathdx:
            y = _rows_kfft_call(KFFT_KMINOR_TARGET, x.reshape(-1, nk), kg,
                                forward=kind == "fftn", scale=scale)
            return y.reshape(x.shape)
        lead = x.reshape(-1, nk).T                        # the plan route is k-leading
        return (_plan_kfft(lead, kg, kind) * scale).T.reshape(x.shape)
    return _kfft


def _sharded(local, mesh, in_specs, out_spec):
    from common.shard_map import shard_map     # see the import-cycle note
    return shard_map(local, mesh=mesh, in_specs=in_specs, out_specs=out_spec,
                     check_vma=False)


def make_kfft_klead(mesh: Mesh, kgrid, spec: P, *, kind: str, norm: str | None) -> Callable:
    """Sharded ``fn(X) -> Y`` over the leading flat-k axis; ``spec`` is the 3-D-form spec."""
    flat = validate_flat_spec(spec, "the input")
    return _sharded(make_local_kfft_klead(mesh, kgrid, kind=kind, norm=norm),
                    mesh, (flat,), flat)


def make_kfft_kminor(mesh: Mesh, kgrid, spec: P, *, kind: str, norm: str | None) -> Callable:
    """Sharded ``fn(X) -> Y`` over the three trailing k axes; they must be replicated."""
    axes = tuple(spec)
    if len(axes) < 3 or any(a is not None for a in axes[-3:]):
        raise ValueError(f"k-minor transform needs the three trailing k axes replicated; got {spec}")
    return _sharded(make_local_kfft_kminor(mesh, kgrid, kind=kind, norm=norm),
                    mesh, (spec,), spec)



def plane_fft_split(n: int) -> tuple[int, int] | None:
    """The Good-Thomas split ``n = n1·n2`` mode 10 transforms an axis of ``n`` points as.

    The most balanced coprime ``n1 >= n2 >= 2`` with ``n1 <= KCONV_AXIS_MAX``
    (shorter cuFFTDx thread FFTs and more lines per pass); ``(n, 1)`` for a
    prime power ``2 <= n <= KCONV_AXIS_MAX`` (one thread FFT); ``None`` when
    neither exists: a prime power above the thread-FFT limit (64, 81, 125,
    128, …) or a prime factor above it.
    """
    n = int(n)
    best = None
    for n1 in range(2, min(n, KCONV_AXIS_MAX + 1)):
        n2 = n // n1
        if n % n1 == 0 and 2 <= n2 <= n1 and math.gcd(n1, n2) == 1 and (best is None or n1 < best[0]):
            best = (n1, n2)
    if best is None and 2 <= n <= KCONV_AXIS_MAX:
        best = (n, 1)
    return best


def _plane_runs(pfc: np.ndarray, n_col: int) -> tuple:
    """The cylinder -> plane zero fill as maximal runs of the flat plane.

    ``(start, stop)`` for cells holding the consecutive columns ``[start,
    stop)``, ``(-1, length)`` for empty cells; concatenating ``F[...,
    start:stop]`` and zero blocks is ``take(F, pfc, mode='fill')`` bit for bit.
    """
    empty = pfc >= n_col
    brk = np.flatnonzero(np.r_[True, (empty[1:] != empty[:-1])
                               | (~empty[1:] & (pfc[1:] != pfc[:-1] + 1))])
    ends = np.r_[brk[1:], pfc.size]
    return tuple((-1, int(e - s)) if empty[s] else (int(pfc[s]), int(pfc[e - 1]) + 1)
                 for s, e in zip(brk, ends))


def _optin_smem_bytes(ordinal: int = 0) -> int | None:
    """The device's opt-in shared memory per block (libcuda attribute 97); None without a driver."""
    try:
        import ctypes
        cu = ctypes.CDLL("libcuda.so.1")
        dev, v = ctypes.c_int(), ctypes.c_int()
        if (cu.cuInit(0) or cu.cuDeviceGet(ctypes.byref(dev), int(ordinal))
                or cu.cuDeviceGetAttribute(ctypes.byref(v), 97, dev)):
            return None
        return int(v.value)
    except Exception:                                         # noqa: BLE001
        return None


def plane_resident_bytes(nb: int, nc: int) -> int:
    """Shared memory one mode-10 block needs for an ``(nb, nc)`` plane, static included.

    The dynamic plane ``16·nb·(nc|1)`` plus the kernel's static tables
    ``live[nb]`` (1 B), ``rowb[nb]`` (4 B) and ``foff[PB]`` (8 B, PB = 1 for
    any plane near the limit), with 16 B of alignment slack.  Mode 10 serves
    the plane only when this fits the device's opt-in shared memory per block;
    the handler's build() applies the same bound.
    """
    nb, nc = int(nb), int(nc)
    return 16 * nb * (nc | 1) + 5 * nb + 8 + 16


def make_plane_fft_gather(mesh: Mesh, plane_from_col, n_col: int, plane_shape) -> Callable:
    """Rank-local ``fn(F) -> Y``: the route-G plane transform read straight from the cylinder.

    ``F (..., n_col)`` c128 holds the occupied cells of an ``(n_b, n_c)``
    plane; ``plane_from_col (n_b·n_c,)`` (host, static) is each cell's column,
    ``n_col`` on an empty one.  ``Y (..., n_b, n_c)`` is::

        Y[..., k_b, k_c] = Σ_{b,c} plane[..., b, c] e^{-2πi (b k_b/n_b + c k_c/n_c)},
        plane = take(F, plane_from_col, axis=-1, mode='fill').reshape(..., n_b, n_c)

    i.e. ``fftn(plane, axes=(-2, -1), norm='backward')``.  ``fn(F, start,
    size)`` transforms the slab ``F[:, start:start+size]`` of ``F (A, S, ...,
    n_col)`` in place (``start`` traced, clamped as ``lax.dynamic_slice``
    clamps; ``size`` static) and returns ``(A, size, ..., n_b, n_c)``.  CUDA: nvidia-mathdx
    mode 10, which never writes the zero plane: the row FFTs run on the
    occupied rows only and gather their cells on load, the column FFTs read
    dead rows as zero, and the plane is stored once.  A plane mode 10 cannot
    serve (an axis with no thread-FFT split, :func:`plane_fft_split`, or a
    plane above the device's opt-in shared memory) takes the XLA route
    (static-run concatenate + cuFFT 2-D), decided here once and announced.
    cpu: the XLA route.  The returned function's ``route`` attribute names
    the one taken (``'mathdx'`` or ``'xla'``).
    """
    nb, nc = (int(v) for v in plane_shape)
    n_col = int(n_col)
    pfc = np.asarray(plane_from_col, dtype=np.int64).reshape(-1)
    if pfc.size != nb * nc:
        raise ValueError(f"plane_from_col has {pfc.size} cells; want n_b·n_c = {nb * nc}")
    if pfc.size and not (0 <= int(pfc.min()) and int(pfc.max()) <= n_col):
        raise ValueError(f"plane_from_col entries must lie in [0, n_col={n_col}] (n_col = "
                         f"empty); got [{int(pfc.min())}, {int(pfc.max())}]")
    runs = _plane_runs(pfc, n_col)

    def _check_c128(F):
        if jnp.dtype(F.dtype) != jnp.dtype(jnp.complex128):
            raise TypeError(f"GATE plane-fft-dtype: got F {F.dtype}; want complex128 (mode 10 and "
                            "its XLA route keep one contract; the kernel is fp64); fix: pass complex128")

    def _xla(F, start=None, size=None):
        _check_c128(F)
        if start is not None:
            F = jax.lax.dynamic_slice_in_dim(F, start, int(size), axis=1)
        z = lambda n: jnp.zeros(F.shape[:-1] + (n,), F.dtype)
        st = jnp.concatenate([F[..., a:e] if a >= 0 else z(e) for a, e in runs], axis=-1)
        from common.fft_helpers import local_fftn3     # see the import-cycle note
        return local_fftn3(st.reshape(F.shape[:-1] + (nb, nc)), axes=(-2, -1), norm="backward")

    _xla.route = "xla"
    if kconv_backend(mesh) != "mathdx":
        return _xla
    from ffi.gate import announce_once
    sb, sc = plane_fft_split(nb), plane_fft_split(nc)
    need, have = plane_resident_bytes(nb, nc), _optin_smem_bytes()
    why = ("an axis has no coprime split into cuFFTDx thread FFTs (<= "
           f"{KCONV_AXIS_MAX})" if sb is None or sc is None else
           f"the resident plane needs {need} B > {have} B of opt-in shared memory"
           if have is None or need > have else "")
    if why:
        announce_once(("plane_fft", nb, nc),
                      f"[plane_fft] plane ({nb},{nc}): XLA route (run concatenate + "
                      f"cuFFT 2-D), not mathdx mode 10: {why}", scope="rank0")
        return _xla
    _require_target(PLANE_FFT_GATHER_TARGET, "CUDA")
    occ = (pfc < n_col).reshape(nb, nc)
    rows = np.flatnonzero(occ.any(axis=1)).astype(np.int32)
    gidx = np.where(occ[rows], pfc.reshape(nb, nc)[rows], -1).astype(np.int32)
    attrs = dict(nb=np.int64(nb), nc=np.int64(nc), b1=np.int64(sb[0]), c1=np.int64(sc[0]),
                 **_mathdx_common())
    announce_once(("plane_fft", nb, nc),
                  f"[plane_fft] plane ({nb},{nc}): mathdx mode 10, splits {sb} x {sc}, "
                  f"{rows.size} of {nb} rows occupied", scope="rank0")

    def _mathdx(F, start=None, size=None):
        _check_c128(F)
        if int(F.shape[-1]) != n_col:
            raise ValueError(f"plane FFT expects F (..., n_col={n_col}); got {F.shape}")
        lead = F.shape[:-1]
        if start is not None:
            if F.ndim < 3 or not 0 < int(size) <= F.shape[1]:
                raise ValueError(f"slab form needs F (A, S, ..., n_col) and 0 < size <= S; got "
                                 f"{F.shape}, size={size}")
            lead = (F.shape[0], int(size)) + F.shape[2:-1]
        out = jax.ShapeDtypeStruct(lead + (nb, nc), F.dtype)
        s0 = jnp.asarray(0 if start is None else start, jnp.int32)
        return jax.ffi.ffi_call(PLANE_FFT_GATHER_TARGET, out)(
            F, jnp.asarray(gidx), jnp.asarray(rows), s0, **attrs)
    _mathdx.route = "mathdx"
    return _mathdx

def make_kconv_klead(mesh: Mesh, kgrid, t_spec: P, w_spec: P, *,
                     norm: str | None = "ortho", mult: float = 1.0) -> KConvStored:
    """The Σ-family k-LEADING convolution, routed by platform.

    ``U = mult · fftn(ifftn(T) · ifftn(W)[:, None, :, None, :])`` for
    ``T``/``U`` ``(nk, a, mx, b, my)`` and ``W`` ``(nk, mx, my)`` c128 (flat k
    leading, specs in the 3-D form).  Returns :class:`KConvStored`.

    CUDA: ``prep`` is the mathdx k-leading transform (``ifftn(W)`` into R space,
    once per W) and ``apply`` the fused mathdx T·W pass, in place on T.  cpu:
    ``prep`` is the identity and ``apply`` the FFTW gw_conv host handler, which
    transforms W itself.
    """
    kg = _check_kgrid(kgrid, kconv_backend(mesh))
    nk = kg[0] * kg[1] * kg[2]
    t_flat = validate_flat_spec(t_spec, "T")
    w_flat = validate_flat_spec(w_spec, "W")
    prep_local, apply_local = _klead_locals(mesh, kg, norm, mult)

    prep_sm = _sharded(prep_local, mesh, (w_flat,), w_flat)
    apply_sm = _sharded(apply_local, mesh, (t_flat, w_flat), t_flat)

    def prep(W):
        _check_complex(W)
        if W.ndim != 3 or int(W.shape[0]) != nk:
            raise ValueError(f"k-leading conv expects W (nk={nk}, mx, my); got {W.shape}")
        return prep_sm(W)

    def apply(T, W_prep):
        _check_complex(T, W_prep)
        if T.ndim != 5 or int(T.shape[0]) != nk:
            raise ValueError(f"k-leading conv expects T (nk={nk}, a, mx, b, my); got {T.shape}")
        return apply_sm(T, W_prep)

    return KConvStored(prep=prep, apply=apply)


def _klead_locals(mesh, kg, norm, mult):
    """Rank-local ``(prep, apply)`` of :func:`make_kconv_klead` for this mesh's backend."""
    nk = kg[0] * kg[1] * kg[2]
    si, sf = ffi_fft_scale("ifftn", norm, nk), ffi_fft_scale("fftn", norm, nk)
    if kconv_backend(mesh) == "mathdx":
        _require_target(KCONV_KLEAD_TARGET, "CUDA")
        prep_local = make_local_kfft_klead(mesh, kg, kind="ifftn", norm=norm)
        attrs = dict(nkx=np.int64(kg[0]), nky=np.int64(kg[1]), nkz=np.int64(kg[2]),
                     scale=np.float64(si * sf * float(mult)))

        def apply_local(t, v_r):
            return jax.ffi.ffi_call(
                KCONV_KLEAD_TARGET, jax.ShapeDtypeStruct(t.shape, t.dtype),
                input_output_aliases={0: 0})(t, v_r, **attrs, **_mathdx_common())
    elif _cpu_test_arm():
        _require_plan_route()
        prep_local = make_local_kfft_klead(mesh, kg, kind="ifftn", norm=norm)

        def apply_local(t, v_r):
            t_r = _plan_kfft(t, kg, "ifftn") * (si * sf * float(mult))
            return _plan_kfft(t_r * v_r[:, None, :, None, :], kg, "fftn")
    else:
        _require_plan_route()
        _require_target(GW_CONV_TARGET, "cpu")

        def prep_local(w):
            return w
        apply_local = _host_gw_conv_local(kg, norm, mult)
    return prep_local, apply_local


def _store_row_map(store_rows, nk: int, label: str) -> tuple[np.ndarray, np.ndarray]:
    """``(rows, kout)``: the stored full-k rows and the (nk,) map k -> output row (-1 = none)."""
    rows = np.asarray(store_rows, dtype=np.int64).reshape(-1)
    if rows.size == 0 or rows.min() < 0 or rows.max() >= nk or np.unique(rows).size != rows.size:
        raise ValueError(f"{label}: store_rows must be distinct full-k rows in [0, {nk}); "
                         f"got {rows.tolist()}")
    kout = np.full(nk, -1, dtype=np.int32)
    kout[rows] = np.arange(rows.size, dtype=np.int32)
    return rows, kout


def make_kconv_klead_unfold(mesh: Mesh, kgrid, tables, *, store_rows, norm: str | None = "ortho",
                            mult: float = 1.0) -> Callable:
    """The Σ k-leading convolution read from the RAW-PARENT Green: ``fn(G, Gt, W_prep) -> U``.

    ``G`` ``(n_parent, mu, ns, nu, ns)`` c128 at ``P(None,'x',None,'y',None)``
    is the centroid-major parent Green (``gw.greens_function_kernel.
    build_G_parents``) and ``Gt`` its transposed partner for the antiunitary
    rows (``None`` when no row is antiunitary); ``tables`` are
    ``symmetry_maps.unfold_load_tables`` of the same plan; ``W_prep`` is
    :func:`make_kconv_klead`'s ``prep(W)`` for the same ``(kgrid, norm, mult)``.
    ``store_rows`` names the full-k rows the caller consumes (the Σ consumers
    pass the plan's ``parent_full_rows``).  Returns ``U``
    ``(len(store_rows), ns, mu, ns, nu)`` at ``P(None,None,'x',None,'y')``,
    equal to ``apply(sigma_conv_operand(unfold_spin_centroid_operator(G, Gt)),
    W_prep)[store_rows]``: the typed unfold, the spin action and the
    spin-major reorder happen on the convolution's load, and every other k
    row is transformed but never stored.  CUDA: nvidia-mathdx mode 7; cpu:
    the service's reference composition, then the plan route and the row
    selection.
    """
    from symmetry_maps import apply_unfold_load_tables_local, local_unfold_load_tables
    kg = _check_kgrid(kgrid, kconv_backend(mesh))
    nk = kg[0] * kg[1] * kg[2]
    if int(tables.row.shape[0]) != nk:
        raise ValueError(f"k-leading unfold conv: tables cover {tables.row.shape[0]} k, grid has {nk}")
    rows, kout = _store_row_map(store_rows, nk, "k-leading unfold conv")
    n_out = int(rows.size)
    ns = int(tables.spin.shape[-1])
    spin_host = np.asarray(tables.spin)
    needs_partner = bool(np.any(np.asarray(tables.trs)))
    mesh_shape = (int(mesh.shape["x"]), int(mesh.shape["y"]))
    if tuple(tables.mesh_shape) != mesh_shape:
        raise ValueError(f"k-leading unfold conv: tables were cut for a {tuple(tables.mesh_shape)} "
                         f"mesh; this mesh is {mesh_shape}")
    si, sf = ffi_fft_scale("ifftn", norm, nk), ffi_fft_scale("fftn", norm, nk)
    if kconv_backend(mesh) == "mathdx":
        _require_target(KCONV_KLEAD_UNFOLD_TARGET, "CUDA")
        attrs = dict(nkx=np.int64(kg[0]), nky=np.int64(kg[1]), nkz=np.int64(kg[2]),
                     scale=np.float64(si * sf * float(mult)), **_mathdx_common())

        def local(g, gt, v_r):
            t = local_unfold_load_tables(tables)
            n_par, mx, _, my, _ = (int(v) for v in g.shape)
            flat = lambda a: a.reshape(n_par, mx * ns, my * ns)
            out = jax.ShapeDtypeStruct((n_out, ns, mx, ns, my), g.dtype)
            return jax.ffi.ffi_call(KCONV_KLEAD_UNFOLD_TARGET, out)(
                flat(g), flat(gt), t.row, t.trs, t.lsrc, t.rsrc, t.mph, t.nph, t.spin,
                jnp.asarray(kout), v_r, **attrs)
    else:
        _, conv_local = _klead_locals(mesh, kg, norm, mult)

        def local(g, gt, v_r):
            t = local_unfold_load_tables(tables)
            n_par, mx, _, my, _ = (int(v) for v in g.shape)
            flat = lambda a: a.reshape(n_par, mx * ns, my * ns)
            O = apply_unfold_load_tables_local(flat(g), flat(gt), t, spin_host)
            return jnp.take(conv_local(jnp.transpose(O, (0, 2, 1, 4, 3)), v_r),
                            jnp.asarray(rows), axis=0)

    g_spec = P(None, "x", None, "y", None)
    sm = _sharded(local, mesh, (g_spec, g_spec, P(None, "x", "y")), P(None, None, "x", None, "y"))

    def apply(G, Gt, W_prep):
        _check_complex(G, W_prep)
        if G.ndim != 5 or int(G.shape[2]) != ns or int(G.shape[4]) != ns:
            raise ValueError(f"k-leading unfold conv expects G (n_parent, mu, {ns}, nu, {ns}); "
                             f"got {G.shape}")
        if Gt is None:
            if needs_partner:
                raise ValueError("k-leading unfold conv: the plan has antiunitary rows, so the "
                                 "transposed parent Green Gt is required")
            Gt = G
        if Gt.shape != G.shape or W_prep.shape != (nk, G.shape[1], G.shape[3]):
            raise ValueError(f"k-leading unfold conv: Gt {Gt.shape} / W_prep {W_prep.shape} do not "
                             f"match G {G.shape} and nk={nk}")
        # The kernel addresses parent row row[k] and local sources below the
        # tables' widths without bounds checks: the operands must be the ones
        # the tables were built for.
        if (int(G.shape[0]) != int(tables.n_parent)
                or int(G.shape[1]) * ns != int(tables.lsrc.shape[1])
                or int(G.shape[3]) * ns != int(tables.rsrc.shape[1])):
            raise ValueError(
                f"k-leading unfold conv: G {G.shape} does not match its tables (n_parent="
                f"{tables.n_parent}, endpoints {tables.lsrc.shape[1]}/{tables.rsrc.shape[1]} "
                f"merged over ns={ns})")
        return sm(G, Gt, W_prep)
    return apply


def _vertex_tables(vertices, ns: int, label: str) -> tuple[np.ndarray, np.ndarray]:
    """``(perm, phase)`` monomial vertices → concatenated perm and phase-code attributes."""
    if not 1 <= len(vertices) <= 4:
        raise ValueError(f"k-conv {label}: 1..4 Lorentz vertices per side, got {len(vertices)}")
    perms = [_check_perm(perm, ns, f"{label} vertex {i}") for i, (perm, _) in enumerate(vertices)]
    codes = [_conv_kpair_phase_codes(phase, ns, f"{label} vertex {i}")
             for i, (_, phase) in enumerate(vertices)]
    return np.concatenate(perms).astype(np.int64), np.concatenate(codes).astype(np.int64)


def make_kconv_lorentz_unfold(mesh: Mesh, kgrid, tables, *, left_vertices, right_vertices,
                              store_rows, norm: str | None = "ortho",
                              mult: float = 1.0) -> Callable:
    """The four-current Σ convolution read from the RAW-PARENT Green: ``fn(G, Gt, V) -> U``.

    ``U[k,a,x,b,y] = mult · fftn( Σ_ij (γ_i ifftn(Ĝ) γ_j†)[a,x,b,y] · ifftn(V)[k,x,i,y,j] )``

    ``Ĝ`` is the full-k Green :func:`make_kconv_klead_unfold` reads from ``G``/``Gt``
    and ``tables`` (the typed unfold, spin action and spin-major order on the
    load); ``left_vertices``/``right_vertices`` are the Lorentz vertices ``γ_i``,
    ``γ_j`` as ``(perm, phase)`` monomial pairs
    (``common.gamma_matrices.gamma_perm_phase``: ``γ[α,β] = phase[α] δ_{β,perm[α]}``),
    and ``V`` ``(nk, mx, nA, my, nB)`` c128 at ``P(None,'x',None,'y',None)`` holds
    the block ``(i, j)`` interaction in k space.  One transform of the Green
    serves every block.  Returns ``U`` ``(len(store_rows), ns, mx, ns, my)``
    at ``P(None,None,'x',None,'y')``, the rows ``store_rows`` of the full-k
    result (:func:`make_kconv_klead_unfold`).

    CUDA: nvidia-mathdx mode 3 on ``V`` then mode 8; each rounds as the XLA
    chain it replaces (the ``norm`` transforms of ``Ĝ`` and ``V``, the vertex
    products, the block sum in ``(i, j)`` order, the forward transform, then
    ``mult``).  cpu: that chain on the service's reference unfold and the plan
    route.
    """
    from symmetry_maps import apply_unfold_load_tables_local, local_unfold_load_tables
    kg = _check_kgrid(kgrid, kconv_backend(mesh))
    nk = kg[0] * kg[1] * kg[2]
    if int(tables.row.shape[0]) != nk:
        raise ValueError(f"k-leading lorentz conv: tables cover {tables.row.shape[0]} k, grid has {nk}")
    rows, kout = _store_row_map(store_rows, nk, "k-leading lorentz conv")
    n_out = int(rows.size)
    ns = int(tables.spin.shape[-1])
    spin_host = np.asarray(tables.spin)
    needs_partner = bool(np.any(np.asarray(tables.trs)))
    mesh_shape = (int(mesh.shape["x"]), int(mesh.shape["y"]))
    if tuple(tables.mesh_shape) != mesh_shape:
        raise ValueError(f"k-leading lorentz conv: tables were cut for a {tuple(tables.mesh_shape)} "
                         f"mesh; this mesh is {mesh_shape}")
    perm_l, phase_l = _vertex_tables(left_vertices, ns, "left")
    perm_r, phase_r = _vertex_tables(right_vertices, ns, "right")
    na, nb = len(left_vertices), len(right_vertices)
    si, sf = ffi_fft_scale("ifftn", norm, nk), ffi_fft_scale("fftn", norm, nk)
    prep_local = make_local_kfft_klead(mesh, kg, kind="ifftn", norm=norm)
    if kconv_backend(mesh) == "mathdx":
        _require_target(KCONV_KLEAD_LORENTZ_TARGET, "CUDA")
        attrs = dict(nkx=np.int64(kg[0]), nky=np.int64(kg[1]), nkz=np.int64(kg[2]),
                     scale_g=np.float64(si), scale_f=np.float64(sf), mult=np.float64(mult),
                     perm_l=perm_l, phase_l=phase_l, perm_r=perm_r, phase_r=phase_r,
                     **_mathdx_common())

        def local(g, gt, v):
            t = local_unfold_load_tables(tables)
            n_par, mx, _, my, _ = (int(d) for d in g.shape)
            flat = lambda a: a.reshape(n_par, mx * ns, my * ns)
            out = jax.ShapeDtypeStruct((n_out, ns, mx, ns, my), g.dtype)
            return jax.ffi.ffi_call(KCONV_KLEAD_LORENTZ_TARGET, out)(
                flat(g), flat(gt), t.row, t.trs, t.lsrc, t.rsrc, t.mph, t.nph, t.spin,
                jnp.asarray(kout), prep_local(v), **attrs)
    else:
        forward_local = make_local_kfft_klead(mesh, kg, kind="fftn", norm=norm)
        quarter = np.asarray([1, 1j, -1, -1j], dtype=np.complex128)
        left = [(perm_l[i * ns:(i + 1) * ns], quarter[phase_l[i * ns:(i + 1) * ns]]) for i in range(na)]
        right = [(perm_r[j * ns:(j + 1) * ns], quarter[phase_r[j * ns:(j + 1) * ns]]) for j in range(nb)]

        def local(g, gt, v):
            t = local_unfold_load_tables(tables)
            n_par, mx, _, my, _ = (int(d) for d in g.shape)
            flat = lambda a: a.reshape(n_par, mx * ns, my * ns)
            O = apply_unfold_load_tables_local(flat(g), flat(gt), t, spin_host)
            green = prep_local(jnp.transpose(O, (0, 2, 1, 4, 3)))
            v_r = prep_local(v)
            total = jnp.zeros_like(green)
            for i, (pl, hl) in enumerate(left):
                for j, (pr, hr) in enumerate(right):
                    # gamma_A on the left spin axis, gamma_B^dagger on the right:
                    # a gather and an exact quarter-turn phase each.
                    value = (jnp.take(green, jnp.asarray(pl), axis=1)
                             * jnp.asarray(hl).reshape(1, ns, 1, 1, 1))
                    value = (jnp.take(value, jnp.asarray(pr), axis=3)
                             * jnp.asarray(np.conj(hr)).reshape(1, 1, 1, ns, 1))
                    total = total + value * v_r[:, None, :, i, None, :, j]
            return jnp.take(forward_local(total) * mult, jnp.asarray(rows), axis=0)

    g_spec = P(None, "x", None, "y", None)
    sm = _sharded(local, mesh, (g_spec, g_spec, g_spec), P(None, None, "x", None, "y"))

    def apply(G, Gt, V):
        _check_complex(G, V)
        if G.ndim != 5 or int(G.shape[2]) != ns or int(G.shape[4]) != ns:
            raise ValueError(f"k-leading lorentz conv expects G (n_parent, mu, {ns}, nu, {ns}); "
                             f"got {G.shape}")
        if Gt is None:
            if needs_partner:
                raise ValueError("k-leading lorentz conv: the plan has antiunitary rows, so the "
                                 "transposed parent Green Gt is required")
            Gt = G
        if Gt.shape != G.shape or V.shape != (nk, G.shape[1], na, G.shape[3], nb):
            raise ValueError(f"k-leading lorentz conv: Gt {Gt.shape} / V {V.shape} do not match "
                             f"G {G.shape}, nk={nk} and ({na}, {nb}) vertices")
        if (int(G.shape[0]) != int(tables.n_parent)
                or int(G.shape[1]) * ns != int(tables.lsrc.shape[1])
                or int(G.shape[3]) * ns != int(tables.rsrc.shape[1])):
            raise ValueError(
                f"k-leading lorentz conv: G {G.shape} does not match its tables (n_parent="
                f"{tables.n_parent}, endpoints {tables.lsrc.shape[1]}/{tables.rsrc.shape[1]} "
                f"merged over ns={ns})")
        return sm(G, Gt, V)
    return apply


def make_kfft_klead_unfold(mesh: Mesh, kgrid, tables, *, norm: str | None = "ortho") -> Callable:
    """An interaction's R-space operand read from its q WEDGE: ``fn(Wp, Wt=None) -> Y``.

    ``Wp`` ``(n_wedge, ml, nl)`` c128 at ``P(None,'x','y')`` holds the
    interaction on the wedge rows (merged endpoints ``ml = mx*n_l``,
    ``nl = my*n_r``; a scalar W has ``n_l = n_r = 1``, a Lorentz block
    ``(mx, nA, my, nB)`` flattened); ``Wt`` is its transposed partner, needed
    only when ``tables`` use the pair-transpose rule on antiunitary rows
    (``tables.conj_trs = 0``).  ``tables`` are
    ``symmetry_maps.unfold_load_tables`` of the q wedge.  Returns ``Y``
    ``(nk, ml, nl)`` at ``P(None,'x','y')``, equal to
    ``make_kconv_klead(...).prep`` of the full-zone interaction
    (``unfold_isdf_operator``, then the endpoint actions): the unfold is the
    transform's load, so the full-zone interaction is never stored.  CUDA:
    nvidia-mathdx mode 9; cpu: the service's reference composition, then the
    prep of the plan route (``ifftn``; the identity on the host-conv arm,
    whose apply transforms W itself).
    """
    from symmetry_maps import apply_unfold_load_tables_local, local_unfold_load_tables
    kg = _check_kgrid(kgrid, kconv_backend(mesh))
    nk = kg[0] * kg[1] * kg[2]
    if int(tables.row.shape[0]) != nk:
        raise ValueError(f"k-leading unfold fft: tables cover {tables.row.shape[0]} k, grid has {nk}")
    spin_l = np.asarray(tables.spin)
    spin_r = spin_l if tables.spin_r is None else np.asarray(tables.spin_r)
    n_l, n_r = int(spin_l.shape[-1]), int(spin_r.shape[-1])
    conj = int(tables.conj_trs)
    needs_partner = bool(np.any(np.asarray(tables.trs))) and not conj
    if kconv_backend(mesh) == "mathdx":
        _require_target(KFFT_KLEAD_UNFOLD_TARGET, "CUDA")
        attrs = dict(nkx=np.int64(kg[0]), nky=np.int64(kg[1]), nkz=np.int64(kg[2]),
                     scale=np.float64(ffi_fft_scale("ifftn", norm, nk)), conj_trs=np.int64(conj),
                     **_mathdx_common())

        def local(w, wt):
            t = local_unfold_load_tables(tables)
            out = jax.ShapeDtypeStruct((nk, int(w.shape[1]), int(w.shape[2])), w.dtype)
            return jax.ffi.ffi_call(KFFT_KLEAD_UNFOLD_TARGET, out)(
                w, wt, t.row, t.trs, t.lsrc, t.rsrc, t.mph, t.nph, t.spin, t.spin_r, **attrs)
    else:
        _require_plan_route()
        prep_local = (make_local_kfft_klead(mesh, kg, kind="ifftn", norm=norm)
                      if _cpu_test_arm() else (lambda o: o))

        def local(w, wt):
            t = local_unfold_load_tables(tables)
            O = apply_unfold_load_tables_local(w, wt, t, spin_l,
                                               None if tables.spin_r is None else spin_r)
            return prep_local(O.reshape(nk, int(w.shape[1]), int(w.shape[2])))

    spec = P(None, "x", "y")
    sm = _sharded(local, mesh, (spec, spec), spec)

    def fn(Wp, Wt=None):
        _check_complex(Wp)
        if Wp.ndim != 3 or int(Wp.shape[1]) % n_l or int(Wp.shape[2]) % n_r:
            raise ValueError(f"k-leading unfold fft expects Wp (n_wedge, mx*{n_l}, my*{n_r}); "
                             f"got {Wp.shape}")
        if int(Wp.shape[0]) != int(tables.n_parent):
            raise ValueError(f"k-leading unfold fft: Wp has {Wp.shape[0]} wedge rows, the "
                             f"tables {tables.n_parent}")
        if Wt is None:
            if needs_partner:
                raise ValueError("k-leading unfold fft: the tables read the transposed partner on "
                                 "antiunitary rows (pair_transpose), so Wt is required")
            Wt = Wp
        return sm(Wp, Wt)
    return fn


def kconv_kminor_out_shape(x_shape, out_layout: int) -> tuple[int, ...]:
    """Output shape of :func:`make_kconv_kminor` for ``X`` of ``x_shape``."""
    d0, d1, d2, d3, d4, nk = (int(v) for v in x_shape)
    if out_layout == 0:
        return (d0, d1, d2, d3, d4, nk)
    if out_layout == 1:
        return (d0, nk, d3, d1, d4, d2)
    raise ValueError(f"out_layout must be 0 or 1, got {out_layout!r}")


def _kminor_out_spec(x_spec: P, out_layout: int) -> P:
    ax = tuple(x_spec)
    return P(*ax) if out_layout == 0 else P(ax[0], ax[5], ax[3], ax[1], ax[4], ax[2])


def make_local_kconv_kminor(mesh: Mesh, kgrid, *, norm: str | None = "ortho",
                            mult: float = 1.0, out_layout: int = 0) -> Callable:
    """Rank-local ``fn(X, K_R) -> U`` of :func:`make_kconv_kminor`, for code already
    inside a shard_map: ``X`` ``(d0, d1, d2, d3, d4, nk)``, ``K_R`` ``(d1, d2, nk)``."""
    if out_layout not in (0, 1):
        raise ValueError(f"out_layout must be 0 or 1, got {out_layout!r}")
    kg = _check_kgrid(kgrid, kconv_backend(mesh))
    nk = kg[0] * kg[1] * kg[2]
    scale = ffi_fft_scale("ifftn", norm, nk) * ffi_fft_scale("fftn", norm, nk) * float(mult)
    if kconv_backend(mesh) == "mathdx":
        _require_target(KCONV_KMINOR_TARGET, "CUDA")
        attrs = dict(nkx=np.int64(kg[0]), nky=np.int64(kg[1]), nkz=np.int64(kg[2]),
                     scale=np.float64(scale), out_layout=np.int64(out_layout))

        def _mathdx(x, k_r):
            _check_complex(x, k_r)
            out = jax.ShapeDtypeStruct(kconv_kminor_out_shape(x.shape, out_layout), x.dtype)
            kw = {"input_output_aliases": {0: 0}} if out_layout == 0 else {}
            return jax.ffi.ffi_call(KCONV_KMINOR_TARGET, out, **kw)(
                x, k_r, **attrs, **_mathdx_common())
        return _mathdx
    _require_plan_route()

    def _plan(x, k_r):
        _check_complex(x, k_r)
        lead = jnp.moveaxis(x, -1, 0)                      # the plan route is k-leading
        u = _plan_kfft(_plan_kfft(lead, kg, "ifftn")
                       * jnp.moveaxis(k_r, -1, 0)[:, None, :, :, None, None], kg, "fftn")
        u = jnp.moveaxis(u * scale, 0, -1)
        return u if out_layout == 0 else jnp.transpose(u, (0, 5, 3, 1, 4, 2))
    return _plan


def make_kconv_kminor(mesh: Mesh, kgrid, x_spec: P, k_spec: P, *,
                      norm: str | None = "ortho", mult: float = 1.0,
                      out_layout: int = 0) -> Callable:
    """The BSE k-MINOR stored-kernel convolution ``fn(X, K_R) -> U``, routed by platform.

    ``U = mult · fftn_k(ifftn_k(X) · K_R[None, :, :, None, None, :])`` for
    ``X`` ``(d0, d1, d2, d3, d4, nk)`` and ``K_R`` ``(d1, d2, nk)`` already in R
    space (the caller made it once with :func:`make_kfft_kminor`).  ``U`` has
    X's layout (``out_layout=0``, in place) or ``(d0, nk, d3, d1, d4, d2)``
    (``out_layout=1``, emitted by the store).  The k axis of both specs must be
    replicated and ``K``'s ``(d1, d2)`` must sit on X's mesh axes.
    """
    if out_layout not in (0, 1):
        raise ValueError(f"out_layout must be 0 or 1, got {out_layout!r}")
    kg = _check_kgrid(kgrid, kconv_backend(mesh))
    nk = kg[0] * kg[1] * kg[2]
    xax, kax = tuple(x_spec), tuple(k_spec)
    if len(xax) != 6 or xax[5] is not None or len(kax) != 3 or kax[2] is not None \
            or (kax[0], kax[1]) != (xax[1], xax[2]):
        raise ValueError(f"k-minor conv wants X (d0..d4, nk) and K (d1, d2, nk) with k "
                         f"replicated and (d1, d2) on the same mesh axes; got {x_spec} / {k_spec}")
    _local = make_local_kconv_kminor(mesh, kg, norm=norm, mult=mult, out_layout=out_layout)
    sm = _sharded(_local, mesh, (x_spec, k_spec), _kminor_out_spec(x_spec, out_layout))

    def conv(X, K_R):
        _check_complex(X, K_R)
        if X.ndim != 6 or K_R.ndim != 3 or int(X.shape[5]) != nk or int(K_R.shape[2]) != nk:
            raise ValueError(f"k-minor conv expects X (d0..d4, {nk}) and K (d1, d2, {nk}); "
                             f"got {X.shape} / {K_R.shape}")
        return sm(X, K_R)

    return conv


# =============================================================================
# The local Fourier plan's CUDA leg (``common.fourier_plan.LocalFourierPlan``)
# =============================================================================
FOURIER_PLAN_TARGET = "lorrax_fourier_plan"


def require_fourier_plan(mesh: Mesh, *, announce: bool = True) -> str:
    """Startup check of ``LocalFourierPlan``'s leg on this mesh; returns it or refuses.

    CUDA: the ``lorrax_fourier_plan`` handler must be in the loaded library
    (the plan's CUDA leg is that one custom call); cpu: the XLA ops, nothing
    to probe.
    """
    from ffi.gate import announce_once, mesh_ffi_platform
    if mesh_ffi_platform(mesh) != "CUDA":
        return "xla"
    _require_target(FOURIER_PLAN_TARGET, "CUDA")
    announce_once(("fourier_plan", "cuda"),
                  f"[fourier_plan] LocalFourierPlan CUDA leg: {FOURIER_PLAN_TARGET} "
                  "(cuBLAS Fourier GEMMs + one cuFFT group)", scope="rank0", emit=announce)
    return "ffi"


def fourier_plan_ffi(x, *, n, kin, kout, in_idx, out_idx, sup_in, sup_out, gemm, scale,
                     order, sign):
    """One ``lorrax_fourier_plan`` custom call over the ``len(n)`` trailing axes
    of ``x`` (row-major in, row-major out; ``cpp/cufft/fourier_plan_cuda_ffi.cc``).

    Every attribute is per transform axis in physical order: full extent
    ``n``, compact extents ``kin``/``kout``, the concatenated supports
    ``in_idx``/``out_idx`` (identity ranges on an axis without one), the 0/1
    flags ``sup_in``/``sup_out``/``gemm``, the axis' jnp.fft ``scale``, and
    ``order``: GEMM axes in execution order with -1 where the FFT group runs.
    """
    d = len(n)
    out = jax.ShapeDtypeStruct(tuple(x.shape[:-d]) + tuple(int(k) for k in kout), x.dtype)
    i64 = lambda v: np.asarray(v, dtype=np.int64)
    return jax.ffi.ffi_call(FOURIER_PLAN_TARGET, out)(
        x, n=i64(n), kin=i64(kin), kout=i64(kout), in_idx=i64(in_idx), out_idx=i64(out_idx),
        sup_in=i64(sup_in), sup_out=i64(sup_out), gemm=i64(gemm),
        scale=np.asarray(scale, dtype=np.float64), order=i64(order), sign=np.int64(sign))
