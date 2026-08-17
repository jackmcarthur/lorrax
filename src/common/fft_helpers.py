import math
from typing import Callable

import jax
import jax.numpy as jnp
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P
from common.shard_map import shard_map


# =============================================================================
# FFT peak-memory query (for memory-model sizing)
# =============================================================================
# ``gflat_memory_model._fft_box_bytes`` needs the per-rank peak HBM a batched
# 3-D FFT will actually take.  Nominal ``N_copies × data_size`` fudge factors
# under-predict badly for mixed-radix boxes (24 = 2³·3, 10 = 2·5) at small
# batch sizes — cuFFT picks different algorithms there, with non-linear
# workspace growth.
#
# The measurement has TWO halves and needs both.  ``compiled.memory_analysis()``
# gives XLA's buffer-assignment peak; the cuFFT *plan workspace* is not in
# buffer assignment at all (jaxlib's FftThunk takes it from a runtime scratch
# allocator), so memory_analysis alone is a systematic low bound.  This
# function compiles the FFT and hands the executable to
# ``runtime.aot_memory.aot_kernel_peak_bytes``, which adds the cuFFT term by
# querying ``cufftMakePlanMany`` on jaxlib's own libcufft.  That module owns
# the cuFFT half; this function owns "which FFT, at which shape/sharding".
#
# Called statically at chooser time (never in a hot loop), so the ~1-2 s
# per-shape compile is amortised over the run and cached per unique shape.

_fft_workspace_cache: dict = {}


def query_fft_peak_bytes(
    *,
    input_shape: tuple[int, ...],
    fft_axes: tuple[int, ...],
    sharding: NamedSharding,
    dtype=jnp.complex128,
) -> int:
    """Per-rank peak HBM bytes of the production 3-D FFT at this shape.

    Compiles the SAME helper production uses — :func:`make_sharded_fftn_3d`,
    a ``shard_map``'d device-local ``jnp.fft.fftn`` — at ``input_shape`` /
    ``sharding``, then returns
    ``runtime.aot_memory.aot_kernel_peak_bytes(...).total``:

        ``temp + argument + output − alias``  (XLA buffer assignment)
      + ``cufftMakePlanMany`` workspace       (invisible to the above)

    i.e. the full per-rank footprint of a standalone FFT jit.  Subtract what
    you already count in other stage terms if you only want the *extra*
    workspace.

    Every path that returns something weaker than that announces itself once
    per process (``runtime.aot_memory.announce_once``): a failed compile
    demotes to a 3×data analytic bound, and a CUDA mesh whose HLO contains no
    parseable fft op — which would silently zero the cuFFT term — is reported
    rather than believed.

    Caches by ``(input_shape, fft_axes, sharding.spec, dtype_str,
    mesh_shape)``, so each unique FFT shape compiles once per process.
    """
    from runtime.aot_memory import aot_kernel_peak_bytes, announce_once

    mesh = sharding.mesh
    key = (tuple(input_shape), tuple(fft_axes),
           str(sharding.spec), jnp.dtype(dtype).str,
           tuple(mesh.axis_names),
           tuple(int(mesh.shape[a]) for a in mesh.axis_names))
    hit = _fft_workspace_cache.get(key)
    if hit is not None:
        return hit

    spec = jax.ShapeDtypeStruct(
        tuple(int(s) for s in input_shape), dtype, sharding=sharding)
    # THE SAME FACTORY PRODUCTION USES.  Every real FFT box in the pipeline
    # (wfn_transforms._local_box_fft, the ζ writer's r→G FFT in
    # gw.isdf_fitting, the flat-k helpers below; zeta_loader's reader-side
    # twin was deleted 2026-08-07) goes through
    # make_sharded_*fftn_3d: one shard_map'd
    # rank-3 ``jnp.fft.fftn`` per rank over its local shard, which XLA:GPU
    # lowers to ONE cuFFT rank-3 plan.  Modelling the per-axis
    # ``make_jittable_local_fftn_3d`` form instead (as this function did
    # before) sizes three rank-1 plans that no production path ever builds.
    local_fftn = make_sharded_fftn_3d(
        mesh, sharding.spec, sharding.spec, axes=tuple(fft_axes), norm=None)
    jit_fft = jax.jit(local_fftn, out_shardings=sharding)

    platform = mesh.devices.flat[0].platform
    on_gpu = platform in ("gpu", "cuda")
    try:
        lowered = jit_fft.lower(spec)
        # The slop factor keeps a deliberately oversized probe box from
        # failing compilation on XLA:GPU's memory limit; it is a GPU-only
        # debug option, so CPU meshes compile plain.
        compiled = lowered.compile(
            compiler_options={"xla_gpu_memory_limit_slop_factor": 10000}
        ) if on_gpu else lowered.compile()
        # Declare the mesh's platform: XLA:CPU also keeps an ``fft`` op in
        # its optimized HLO (measured, jax 0.9.1), so the microservice cannot
        # infer from the HLO alone whether a zero cuFFT term is a fact or a
        # missing measurement.
        peak = aot_kernel_peak_bytes(compiled, platform=platform)
    except Exception as exc:
        # Unusual (e.g. called before JAX has a backend, or a shape the mesh
        # cannot shard).  Demote to an over-conservative 3×data bound — and
        # SAY SO, from this rank.  The pre-2026-07-30 code claimed in a
        # comment that this was "logged so the caller notices" while making
        # no log call at all.
        elem = jnp.dtype(dtype).itemsize
        total_elems = 1
        for s in input_shape:
            total_elems *= int(s)
        n_devs = 1
        for a in mesh.axis_names:
            n_devs *= int(mesh.shape[a])
        fallback = 3 * total_elems * elem // max(1, n_devs)
        announce_once(
            f"fft-peak-compile-failed:{key}",
            f"FFT peak query could NOT compile {tuple(input_shape)} "
            f"{jnp.dtype(dtype).name} on spec {sharding.spec} "
            f"({type(exc).__name__}: {exc}).  Falling back to a 3×data "
            f"analytic bound ({fallback/1e9:.2f} GB/rank) that does NOT "
            f"include cuFFT plan workspace")
        _fft_workspace_cache[key] = fallback
        return fallback

    if not peak.fft_specs:
        # We just compiled an FFT, so its op must be in the HLO — on BOTH
        # platforms (measured, jax 0.9.1: XLA:CPU keeps the fft op too).  If
        # it is not, the parser has gone blind and on a CUDA mesh that means
        # the cuFFT term is 0 by omission rather than by measurement — the
        # exact under-prediction this path exists to remove.
        announce_once(
            f"fft-peak-no-hlo-fft:{key}",
            f"FFT peak query compiled {tuple(input_shape)} on a {platform!r} "
            f"mesh but the optimized HLO exposes NO fft op.  On a CUDA mesh "
            f"that silently zeroes the cuFFT plan-workspace term — XLA has "
            f"probably changed how it emits FFTs; see "
            f"runtime.aot_memory._FFT_OP_RE")
    total = int(peak.total)
    _fft_workspace_cache[key] = total
    return total


def compute_block_size_for_2d_cholesky(n_rmu: int, Pr: int, Pc: int) -> tuple[int, int]:
    """
    Compute block size for 2D blocked Cholesky that satisfies distribution constraints.

    Constraints (fundamental to 2D blocked algorithms):
        - n_rmu % block_size == 0  (matrix divides into whole tiles)
        - J % Pr == 0              (tile rows distribute evenly on X-axis)
        - J % Pc == 0              (tile cols distribute evenly on Y-axis)

    Where J = n_rmu / block_size is the number of tiles per dimension.

    The simplest solution: J = lcm(Pr, Pc), giving block_size = n_rmu / J.
    If n_rmu doesn't divide evenly, we try multiples of lcm(Pr, Pc).

    Args:
        n_rmu: Matrix dimension (number of ISDF centroids)
        Pr: Number of devices on X-axis
        Pc: Number of devices on Y-axis

    Returns:
        (block_size, J) tuple

    Raises:
        ValueError: If no valid block size exists (n_rmu incompatible with mesh)
    """
    target_J = math.lcm(Pr, Pc)

    if n_rmu % target_J == 0:
        block_size = n_rmu // target_J
        return block_size, target_J

    # Try multiples of lcm(Pr, Pc)
    for j_mult in range(2, 20):
        J = target_J * j_mult
        if n_rmu % J == 0:
            block_size = n_rmu // J
            return block_size, J

    # Last resort: find any valid block size
    for b in range(n_rmu, 0, -1):
        if n_rmu % b == 0:
            J = n_rmu // b
            if J % Pr == 0 and J % Pc == 0:
                return b, J

    raise ValueError(
        f"No valid block size for n_rmu={n_rmu} with mesh {Pr}×{Pc}. "
        f"n_rmu should be divisible by lcm({Pr},{Pc})={target_J} or a multiple thereof."
    )


# ============================================================================
# shard_map based FFT - runs FFT independently on each device's local data
# See: https://docs.jax.dev/en/latest/notebooks/shard_map.html
#
# ONE local-FFT form, deliberately.  A second, per-axis
# ``custom_partitioning`` form (``make_jittable_local_{i,}fftn_3d``, three
# chained rank-1 FFTs) lived here until 2026-07-30 with no production caller:
# its only user was ``query_fft_peak_bytes``, i.e. the memory model was
# sizing three rank-1 cuFFT plans that nothing in the pipeline ever builds.
# Recover it from git history if a per-axis form is ever wanted again; do not
# reintroduce it as a modelling-only path.
# ============================================================================


def local_ifftn3(x_local, *, axes: tuple[int, ...] = (-3, -2, -1), norm: str | None = None):
    """Device-local N-D IFFT — the inner kernel of :func:`make_sharded_ifftn_3d`.

    Call this DIRECTLY from code that is already inside a ``shard_map`` (shard_map
    cannot nest): the operand is a device-local shard whose FFT axes are
    replicated, so a plain ``jnp.fft.ifftn`` runs entirely on-device.  The
    ``make_sharded_ifftn_3d`` factory below is just this kernel wrapped in a
    ``shard_map`` for auto-partitioned callers.  ONE source for the local FFT.

    Do NOT call it eagerly on a (μ,ν)-sharded global array: outside a
    shard_map, jax gathers the full operand onto every rank — an N_μ²-class
    tile, forbidden by the scaling doctrine (audit P0-4).
    ``tests/test_fft_shardmap_context.py`` gates this by AST for src/bse and
    src/gw: every call site's enclosing-function chain must construct a
    shard_map (or be a ratcheted, documented single-device exception).
    """
    return jnp.fft.ifftn(x_local, axes=axes, norm=norm)


def local_fftn3(x_local, *, axes: tuple[int, ...] = (-3, -2, -1), norm: str | None = None):
    """Device-local N-D forward FFT — inner kernel of :func:`make_sharded_fftn_3d`.

    Forward counterpart of :func:`local_ifftn3`; call directly from inside a
    ``shard_map``.
    """
    return jnp.fft.fftn(x_local, axes=axes, norm=norm)


def make_sharded_ifftn_3d(
	mesh: Mesh,
	in_spec: P,
	out_spec: P,
	*,
	norm: str | None = None,
	axes: tuple[int, int, int] = (-3, -2, -1),
):
    """
    Uses shard_map to run FFT independently on each device's local data.
    The FFT axes (last 3) must NOT be sharded - only batch dims can be sharded.
    Args:
        mesh: The device mesh
        in_spec: PartitionSpec for input (e.g., P(None, ('x','y'), None, None, None, None))
        out_spec: PartitionSpec for output (same as in_spec for FFT)

    Returns:
        A function that performs 3D IFFT on sharded data
    """
    def _wrap(x_local):
        return local_ifftn3(x_local, axes=axes, norm=norm)

    return shard_map(_wrap, mesh=mesh, in_specs=(in_spec,), out_specs=out_spec)

# ---------------------------------------------------------------------------
# The DONATED transform, memoised — one program per (mesh, spec, axes, norm)
# ---------------------------------------------------------------------------
# ``make_sharded_ifftn_3d`` is a FACTORY: it returns a fresh ``shard_map``
# closure every call and carries no cache.  Wrapping that fresh closure in a
# fresh ``jax.jit`` — which is what every donated-W call site did — hands jax a
# wrapper object whose dispatch cache is empty, so a byte-identical program is
# re-traced, re-lowered and re-probed against the persistent compile cache on
# every call.  It is the same defect ``bse_davidson_helpers`` fixed for the
# exact-diagonal build (PRECOND_BUILD_FREE.md §3.1, §7.2), and it is measured
# at ~20-23 ms per call against ~1.0 ms of execution on the Si 4x4x4 record
# deck at P=4 (FIX_construction_defects.md §1).
#
# The memo below keys on everything that determines the PROGRAM — the mesh, the
# partition spec, the transformed axes and the norm convention.  It deliberately
# does NOT key on shape or dtype: those are jax's own dispatch-cache business,
# so one memo entry serves every payload shape a run puts through the same
# transform.  ``Mesh`` and ``PartitionSpec`` are both hashable by value, so two
# call sites that ask for the same transform share one program.
#
# WHY A SEPARATE ACCESSOR AND NOT A MEMO ON THE FACTORY.  Ten call sites in the
# tree build sharded transforms through the factory, and several close the
# returned callable into larger objects.  Memoising the factory itself would
# change the OBJECT IDENTITY those call sites see, and identity is load-bearing
# in this tree (``bse_feast._GMRES_SOLVER_CACHE`` keys on ``id(matvec)``).  A
# new accessor changes nothing for anyone who does not call it.
_DONATED_IFFTN_3D: dict[tuple, Callable] = {}


def get_donated_ifftn_3d(
    mesh: Mesh,
    spec: P,
    *,
    axes: tuple[int, int, int] = (-3, -2, -1),
    norm: str | None = None,
) -> Callable:
    """The sharded 3-D inverse FFT as a jitted, INPUT-DONATING program.

    Returns the SAME jitted callable for the same ``(mesh, spec, axes, norm)``,
    so the program is constructed once per process instead of once per call.

    Donation is the point of the separate accessor.  The W-transform call sites
    want XLA to alias ``W_R`` onto ``W_q``'s buffer — at production
    ``mu = 10015 / P = 64`` the un-aliased form costs 2 x 404 MB per rank — and
    donation only works from a top-level dispatch boundary, which is what this
    is.  Callers must drop their own reference to the operand right after the
    call; the returned array is the only live copy.

    Parameters
    ----------
    mesh, spec : the device mesh and the operand's partition spec.  The
        transformed ``axes`` must be REPLICATED in ``spec`` — the transform is
        device-local inside a ``shard_map``.
    axes, norm : forwarded to :func:`local_ifftn3`, and part of the memo key
        because both change the emitted program.
    """
    key = (mesh, spec, tuple(axes), norm)
    hit = _DONATED_IFFTN_3D.get(key)
    if hit is None:
        hit = jax.jit(
            make_sharded_ifftn_3d(mesh, spec, spec, axes=tuple(axes),
                                  norm=norm),
            donate_argnums=(0,))
        _DONATED_IFFTN_3D[key] = hit
    return hit


def make_sharded_fftn_3d(
	mesh: Mesh,
	in_spec: P,
	out_spec: P,
	*,
	norm: str | None = None,
	axes: tuple[int, int, int] = (-3, -2, -1),
):
    """
    shard_map local FFT (forward).

    This is the forward-FFT counterpart to make_sharded_ifftn_3d.
    """
    def _wrap(x_local):
        return local_fftn3(x_local, axes=axes, norm=norm)

    return shard_map(_wrap, mesh=mesh, in_specs=(in_spec,), out_specs=out_spec)


# ============================================================================
# FFI backend for the flat-k helpers — REQUIRED (decisions.md 2026-08-01)
# ============================================================================
# The service itself lives in ``src/ffi/fft.py`` (Python) + ``src/ffi/cpp/
# mklfft`` and ``src/ffi/cpp/cufft`` (handlers).  THIS file used to carry a
# second, drifting copy of the gate and both bodies (delegated 2026-07-30),
# and then a gated XLA twin of the flat-k transform (DELETED 2026-08-01
# under the FFI-required ruling: where a certified FFI path exists, the
# native-JAX duplicate is not maintained).
#
# What the service is: the flat-k batched 3-D FFTs dispatched to the platform
# FFI library — the FFTW3 ABI on cpu meshes (NOT DFTI; those calls were deleted
# 2026-08-05, see ffi/fft.py:138), cuFFT with
# the advanced data layout (cufftPlanMany64) on CUDA meshes.  Both libraries
# register the SAME target names, so the call sites are platform-agnostic.
# WHY: XLA's fft custom-call wants the transformed axes minor-most, so every
# dot(k-major) <-> fft(k-minor) boundary in the Σ τ kernel pays a full
# transpose of the ~398 MB/rank μ² tile — 65% of the STAGED τ DISPATCH at
# nb=128/P=64, BEFORE this service existed (the line said "of sigma.exec"
# until 2026-08-11; see the denominator note in ffi/fft.py).  Today the FFT's
# share of that dispatch is 16.1% / 60.5% / 84.9% at 9 / 64 / 216 k-points
# (P=4, BFC@0.85, 2026-08-11) — it is governed by nk, so there is no single
# "today" number and the 9-k fixture's 0.07%-of-wall does not generalise.
# Stride descriptors read the dot-layout tile where it lies, so the transposes
# disappear instead of moving.  Contract: ``docs/dev/flat_k_fft_service.md``.
#
# What stays here: the OWNER RULE that these helpers are the single FFT entry
# point (``make_flat_k_fft`` below is still the only door), and the XLA
# ``make_sharded_*fftn_3d`` / ``local_*fftn3`` layer above — those serve the
# shard_map-INTERIOR call sites (isdf/core, wfn_transforms, BSE) that have
# no FFI route, which the ruling explicitly keeps.
# ============================================================================

from ffi.mklfft import (  # noqa: E402  (re-export: see the block above)
    GATE,
    fft_ffi_enabled,
    make_gw_conv_ffi as _make_gw_conv_ffi,
    make_flat_k_fft_ffi as _make_flat_k_fft_ffi,
)


# ============================================================================
# THE FUSED-CONV FAMILY at the factory seam
# ============================================================================
# Three entries, ONE contract:
#
#     U = scale · FFT_k( IFFT_k(X) · K )
#
# — both transforms, the broadcast multiply against a STORED kernel, and every
# norm factor as a single constant, in ONE FFI call per rank, so the R-space
# intermediate never materialises.  The members differ ONLY in where the
# caller's tile already keeps its k axis, because that is what decides which
# memory layout the one kernel has to read:
#
#     make_flat_k_gw_conv     k LEADING     lorrax_mklfft_gw_conv     cpu+CUDA
#     make_fused_conv_klead   k LEADING     lorrax_cufft_conv_klead  CUDA
#     make_fused_conv_kminor  k MINOR-most  lorrax_cufft_conv_kminor  CUDA
#
# Pick by resident layout; callers do not independently transpose to reach a
# different member.  The measured Sigma factory is the one owned exception:
# its conditional half-absorbed form packs once internally and emits the
# inverse permutation from the store.  The family contract, measurements and
# the reason k-minor takes its kernel already in R space live in ONE place —
# ``ffi/fft.py``'s module docstring.  This block is the seam, not a second copy.
# ============================================================================


def make_flat_k_gw_conv(
    mesh: Mesh,
    kgrid: tuple[int, int, int],
    g_spec: P,
    v_spec: P,
    *,
    norm: str | None = 'ortho',
    mult: float = 1.0,
) -> Callable:
    """Fused conv, **k-LEADING** ``fn(G_flat, W_flat) -> sigma_flat``:

        sigma = fftn( ifftn(G) * ifftn(W)[:, None, :, None, :] * mult )

    with all three transforms + the broadcast multiply in ONE FFI call per
    rank, so the R-space G tile never materializes.  Sigma-family layout
    contract only; the plain helpers remain the entry point for everything
    else.  Sibling: :func:`make_fused_conv_kminor`, same expression for a
    caller whose k axis is already minor-most.  Implementation:
    :func:`ffi.fft.make_gw_conv_ffi`.
    """
    return _make_gw_conv_ffi(mesh, kgrid, g_spec, v_spec,
                             norm=norm, mult=mult)


def make_fused_conv_kminor(
    mesh: Mesh,
    kgrid: tuple[int, int, int],
    x_spec: P,
    k_spec: P,
    *,
    norm: str | None = 'ortho',
    mult: float = 1.0,
    out_layout: int = 0,
) -> Callable:
    """Fused conv, **k-MINOR** ``fn(X, K) -> U``:

        U = scale · fftn_k( ifftn_k(X) · K[None, :, :, None, None, :] )

    ``X`` ``(d0,d1,d2,d3,d4,nk)``, ``K`` ``(d1,d2,nk)`` ALREADY in R space
    (this member multiplies its stored kernel, it does not transform it — the
    caller builds that once and reuses it; see ``ffi/fft.py``).  ONE FFI call:
    both transforms, the broadcast multiply, both norm factors as a single
    constant, and — at ``out_layout=1`` — a consumer's
    ``(d0,nk,d3,d1,d4,d2)`` permutation emitted from the STORE, so a
    downstream layout costs nothing extra.

    CUDA only, and **default OFF** (``LORRAX_CONV_KMINOR_FFI``): a new
    hand-written kernel, exercised at P=1 only.  Sibling:
    :func:`make_flat_k_gw_conv`.  Implementation:
    :func:`ffi.fft.make_conv_kminor_ffi`.
    """
    from ffi.fft import make_conv_kminor_ffi as _impl
    return _impl(mesh, kgrid, x_spec, k_spec,
                 norm=norm, mult=mult, out_layout=out_layout)


def make_fused_conv_klead(
    mesh: Mesh,
    kgrid: tuple[int, int, int],
    g_spec: P,
    v_spec: P,
    *,
    norm: str | None = 'ortho',
    mult: float = 1.0,
) -> Callable:
    """Direct fused conv in Sigma's native **k-LEADING** layout.

    Same public shapes as :func:`make_flat_k_gw_conv`.  The CUDA-only handler
    keeps the public T/W/U arrays k-leading, assembles T and W row pairs in
    shared memory, performs the two inverse transforms, multiply, and forward
    transform in one traversal, and emits the result in Sigma's native
    k-leading layout from its store.
    It is an accelerator behind ``LORRAX_CONV_KLEAD_FFI``; callers retain the
    plan-based member as the off/unsupported path.
    """
    from ffi.fft import make_conv_klead_ffi as _impl
    return _impl(mesh, kgrid, g_spec, v_spec, norm=norm, mult=mult)



# ============================================================================
# Flat-k FFT helpers — callers operate on (nk, *trail) arrays everywhere and
# the k-grid 3D form only exists in the spec vocabulary, matching the
# "flatten kx/ky/kz except inside the FFT" convention used across the GW
# pipeline (w_isdf chi0, ppm_sigma, gw_jax static COHSEX, isdf_fitting
# CCT/ZCT).
#
# Backend: the platform FFI handler, unconditionally (MKL DFTI on cpu,
# cuFFT strided on CUDA — see the block above) — ``(nk, *trail) ->
# (nk, *trail)``, k-major end to end, no 3-D reshape, no layout anchoring.
# The gated XLA twin (reshape -> make_sharded_*fftn_3d -> reshape) was
# DELETED 2026-08-01 (decisions.md: FFI backends are required, not
# optional); LORRAX_FFT_FFI=0 therefore refuses here rather than silently
# selecting a path that no longer exists.  Recover the XLA arm from git
# history for a debugging build.
# ============================================================================


def make_flat_k_fft(
    mesh: Mesh,
    kgrid: tuple[int, int, int],
    spec: P,
    *,
    kind: str,
    norm: str | None = 'ortho',
    out_spec: P | None = None,
) -> Callable:
    """Return a flat-k FFT: ``(nk, *trail) -> (nk, *trail)``.

    ``spec`` is the ``PartitionSpec`` on the 3-D form
    ``(nkx, nky, nkz, *trail)``.  The three leading k-axes must be
    replicated (``None``) so the batched per-rank FFT sees the full FFT
    axis on every device.  ``out_spec`` must equal ``spec`` (the FFI
    backend implements no post-FFT reshard).

    ``kind='ifftn'`` or ``'fftn'`` selects the direction.  ``norm``
    follows ``jnp.fft.*`` ('ortho', 'forward', 'backward' / None).

    Always the platform FFI backend (announce-or-refuse inside; the
    factory-time ``fft_ffi_enabled()`` read stays in every consumer's
    kernel cache key — see ppm_tau_kernel).
    """
    if not fft_ffi_enabled():
        raise RuntimeError(GATE.off_refuse_msg)
    return _make_flat_k_fft_ffi(mesh, kgrid, spec, kind=kind,
                                norm=norm, out_spec=out_spec)


def make_flat_k_ifftn(
    mesh: Mesh,
    kgrid: tuple[int, int, int],
    spec: P,
    *,
    norm: str | None = 'ortho',
    out_spec: P | None = None,
) -> Callable:
    """Flat-k IFFT ``(nk, *trail) -> (nk, *trail)``.  See :func:`make_flat_k_fft`."""
    return make_flat_k_fft(mesh, kgrid, spec, kind='ifftn',
                           norm=norm, out_spec=out_spec)


def make_flat_k_fftn(
    mesh: Mesh,
    kgrid: tuple[int, int, int],
    spec: P,
    *,
    norm: str | None = 'ortho',
    out_spec: P | None = None,
) -> Callable:
    """Flat-k FFT ``(nk, *trail) -> (nk, *trail)``.  See :func:`make_flat_k_fft`."""
    return make_flat_k_fft(mesh, kgrid, spec, kind='fftn',
                           norm=norm, out_spec=out_spec)
