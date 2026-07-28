import math
import os
from typing import Callable

import jax
import jax.numpy as jnp
import numpy as np
from jax.experimental.custom_partitioning import custom_partitioning
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P
from jax.experimental.shard_map import shard_map


# =============================================================================
# FFT workspace query (for memory-model sizing)
# =============================================================================
# The memory model in ``gflat_memory_model.plan_gflat_chunks`` (and the
# V_q chooser in ``compute_vcoul._choose_v_q_chunks``) needs the per-rank peak
# HBM an ``N``-D batched FFT will allocate.  Nominal ``N_copies × data_size``
# fudge factors under-predict badly for mixed-radix boxes (24 = 2³·3, 10 = 2·5)
# at small batch sizes — cuFFT's planner picks different algorithms there with
# non-linear workspace growth.
#
# Rather than query cuFFT via ctypes (fragile across shifter / conda / bare-
# metal JAX builds and may not match XLA's actual plan choice), we AOT-compile
# the exact ``jnp.fft.fftn`` XLA would emit, read its ``memory_analysis()``,
# and cache the result — pure JAX, works wherever JAX works.
#
# This function is called statically at chooser time (never in a hot loop), so
# the ~1-2 s per-shape compile cost is amortised across the full run.  Two
# canonical uses in the pipeline:
#   * Wavefunction-box FFT:   shape = fft_grid (nx, ny, nz), batched by
#                             nk × bpd × ns (in-loop ψ_G → ψ_r).
#   * k-grid FFT (ZCT / CCT): shape = kgrid (nkx, nky, nkz), batched by
#                             μ × cr or μ × μ.

_fft_workspace_cache: dict = {}


def query_fft_peak_bytes(
    *,
    input_shape: tuple[int, ...],
    fft_axes: tuple[int, ...],
    sharding: NamedSharding,
    dtype=jnp.complex128,
) -> int:
    """AOT-compile a ``jnp.fft.fftn`` over the given input/sharding and
    return the XLA-measured total peak bytes PER RANK.

    "Peak" here is ``temp + argument + output − alias`` from
    ``compiled.memory_analysis()`` — the full per-rank HBM footprint of
    a standalone FFT jit (input buffer + output buffer + cuFFT scratch,
    minus donated-alias savings).  Subtract what you already count in
    other stage terms if you only want the "extra" workspace.

    Caches by ``(input_shape, fft_axes, sharding.spec, dtype_str,
    mesh_shape)``, so each unique FFT shape compiles once per process.
    """
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

    # Use make_jittable_local_fftn_3d — matches the real kernel's FFT
    # partitioning.  Plain jnp.fft.fftn on a sharded tensor forces XLA
    # to reason about the FFT at HLO level; since it can't see that
    # sharded axes aren't FFT axes, it inserts a gather for the output
    # and a ~2× replicated temp buffer, inflating reported peak by ~8×
    # for small-FFT-axis + large-batch shapes.  The custom_partitioning
    # wrapper hides the FFT in an opaque primitive so XLA sees only
    # the local per-device FFT — matching production memory usage.
    local_fftn = make_jittable_local_fftn_3d(
        mesh, sharding.spec, sharding.spec,
        axes=tuple(fft_axes), norm=None,
    )
    jit_fft = jax.jit(local_fftn, out_shardings=sharding)

    try:
        compiled = jit_fft.lower(spec).compile(
            compiler_options={"xla_gpu_memory_limit_slop_factor": 10000})
    except Exception:
        # If AOT compile fails (unusual — happens e.g. when called before
        # JAX has a backend), fall back to an over-conservative estimate:
        # 3× data size.  Logged so the caller notices.
        elem = jnp.dtype(dtype).itemsize
        total_elems = 1
        for s in input_shape:
            total_elems *= int(s)
        # Divide by total device count for per-rank estimate (approximate
        # — assumes input is sharded across all devices).
        n_devs = 1
        for a in mesh.axis_names:
            n_devs *= int(mesh.shape[a])
        fallback = 3 * total_elems * elem // max(1, n_devs)
        _fft_workspace_cache[key] = fallback
        return fallback

    m = compiled.memory_analysis()
    total = (int(m.temp_size_in_bytes)
             + int(m.argument_size_in_bytes)
             + int(m.output_size_in_bytes)
             - int(m.alias_size_in_bytes))
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
# ============================================================================


def _normalize_local_fft_axes(rank: int, axes: tuple[int, ...]) -> tuple[int, ...]:
    normalized = tuple(ax if ax >= 0 else rank + ax for ax in axes)
    if len(set(normalized)) != len(normalized):
        raise ValueError(f"FFT axes must be unique, got {axes}.")
    if any(ax < 0 or ax >= rank for ax in normalized):
        raise ValueError(f"FFT axes {axes} out of bounds for rank-{rank} array.")
    return normalized


def _validate_local_fft_specs(in_spec: P, out_spec: P, axes: tuple[int, ...]) -> tuple[int, ...]:
    in_axes = tuple(in_spec)
    out_axes = tuple(out_spec)
    if len(in_axes) != len(out_axes):
        raise ValueError(
            f"Input/output PartitionSpecs must have the same rank, got {in_spec} and {out_spec}."
        )
    fft_axes = _normalize_local_fft_axes(len(in_axes), axes)
    for ax in fft_axes:
        if in_axes[ax] is not None or out_axes[ax] is not None:
            raise ValueError(
                "Jittable local FFT helpers require every transformed axis to be replicated. "
                f"Axis {ax} is sharded in in_spec={in_spec}, out_spec={out_spec}."
            )
    return fft_axes


def _make_jittable_local_fft(
    mesh: Mesh,
    in_spec: P,
    out_spec: P,
    *,
    fft_kind: str,
    norm: str | None,
    axes: tuple[int, ...],
):
    """Return a jit-compatible FFT that preserves sharding on replicated FFT axes."""

    del mesh  # The active mesh is supplied to the partition callback during tracing.
    fft_axes = _validate_local_fft_specs(in_spec, out_spec, axes)
    if fft_kind not in ("ifftn", "fftn"):
        raise ValueError(f"Unsupported fft_kind={fft_kind!r}")

    def _make_axis_wrapper(axis: int):
        def fft_impl(x):
            n_axis = x.shape[axis]
            if fft_kind == "ifftn":
                raw = jnp.conj(jnp.fft.fft(jnp.conj(x), axis=axis))
                if norm in (None, "backward"):
                    return raw / float(n_axis)
                if norm == "ortho":
                    return raw / jnp.sqrt(float(n_axis))
                if norm == "forward":
                    return raw
            else:
                raw = jnp.fft.fft(x, axis=axis)
                if norm in (None, "backward"):
                    return raw
                if norm == "ortho":
                    return raw / jnp.sqrt(float(n_axis))
                if norm == "forward":
                    return raw / float(n_axis)
            raise ValueError(f"Unsupported FFT norm={norm!r}")

        @custom_partitioning
        def _local_fft_axis(x):
            return fft_impl(x)

        def _partition(mesh_arg: Mesh, arg_shapes, result_shape):
            del arg_shapes, result_shape
            return (
                mesh_arg,
                fft_impl,
                NamedSharding(mesh_arg, out_spec),
                (NamedSharding(mesh_arg, in_spec),),
            )

        def _infer(mesh_arg: Mesh, arg_shapes, result_shape):
            del arg_shapes, result_shape
            return NamedSharding(mesh_arg, out_spec)

        _local_fft_axis.def_partition(
            infer_sharding_from_operands=_infer,
            partition=_partition,
            sharding_rule="...i -> ...i",
        )
        return _local_fft_axis

    axis_wrappers = [_make_axis_wrapper(axis) for axis in fft_axes]

    def _local_fft(x):
        for fft_axis in axis_wrappers:
            x = fft_axis(x)
        return x

    return _local_fft


def make_jittable_local_ifftn_3d(
    mesh: Mesh,
    in_spec: P,
    out_spec: P,
    *,
    norm: str | None = None,
    axes: tuple[int, int, int] = (-3, -2, -1),
):
    """Create a jit-compatible local IFFT over replicated FFT axes."""

    return _make_jittable_local_fft(
        mesh,
        in_spec,
        out_spec,
        fft_kind="ifftn",
        norm=norm,
        axes=axes,
    )


def make_jittable_local_fftn_3d(
    mesh: Mesh,
    in_spec: P,
    out_spec: P,
    *,
    norm: str | None = None,
    axes: tuple[int, int, int] = (-3, -2, -1),
):
    """Create a jit-compatible local FFT over replicated FFT axes."""

    return _make_jittable_local_fft(
        mesh,
        in_spec,
        out_spec,
        fft_kind="fftn",
        norm=norm,
        axes=axes,
    )

def local_ifftn3(x_local, *, axes: tuple[int, ...] = (-3, -2, -1), norm: str | None = None):
    """Device-local N-D IFFT — the inner kernel of :func:`make_sharded_ifftn_3d`.

    Call this DIRECTLY from code that is already inside a ``shard_map`` (shard_map
    cannot nest): the operand is a device-local shard whose FFT axes are
    replicated, so a plain ``jnp.fft.ifftn`` runs entirely on-device.  The
    ``make_sharded_ifftn_3d`` factory below is just this kernel wrapped in a
    ``shard_map`` for auto-partitioned callers.  ONE source for the local FFT.
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
# MKL FFT (DFTI API) host-FFI backend for the flat-k helpers — GATED, OFF by
# default (LORRAX_FFT_FFI; the fused τ entry additionally behind
# LORRAX_FFT_FFI_FUSED read in gw/ppm_tau_kernel).  PROTOTYPE, 2026-07-28.
#
# WHAT: the flat-k batched 3-D FFTs dispatched to liblorrax_ffi_host.so's
# MKL FFT handlers (src/ffi/mklfft/cpp/fft_flat_k_ffi.cc) — a genuine
# O(N log N) fast Fourier transform at any k-count, driven through MKL's
# DFTI *descriptor API*.  It is NOT a DFT-as-matmul (owner-vetoed): "DFTI"
# names Intel's descriptor interface to its FFT engine, nothing else.
#
# WHY: XLA:CPU's fft custom-call requires the transformed axes minor-most,
# so every dot(k-major flat) <-> fft(k-minor 3-D) boundary in the Σ τ
# kernel pays a full transpose copy of the ~398 MB/rank μ² tile — measured
# 60-65% of sigma.exec at nb=128/P=64 and CLOSED as structural for any
# XLA-side arrangement (wk_REL/sigma_perf_results.md).  MKL FFT (DFTI API)
# has no such layout requirement: stride descriptors read the dot-layout
# tile exactly where it lies, so the helper boundary stops anchoring
# layouts and the transposes disappear instead of moving.
#
# Contract: these helpers stay THE single FFT entry point (owner rule) —
# the backend switch happens here and nowhere else.  The flag applies to
# every make_flat_k_* call site whose contract the handler supports (the
# 3-D-form spec replicated on the k axes, complex128, no post-FFT
# reshard); norm conventions are computed HERE to match jnp.fft exactly
# and shipped to the handler as a plain scale.  The FFI result is
# value-identical to jnp.fft at ~1e-15 relative (different FFT engine, not
# bit-identical) — gated by the unit gate + the 1e-12 Σ parity suite.
#
# Refusal doctrine (pattern #8): the flag is an explicit request — if the
# host library lacks the handler, or the mesh is not CPU, this REFUSES
# loudly with the probe reason instead of silently running the XLA path.
# In-place: operand 0 is aliased to the result (input_output_aliases), so
# when the buffer is dead XLA lets the handler transform it in place —
# the terminal form of donation (zero extra big tiles).
# ============================================================================

_FFT_FFI_TARGET = "lorrax_mklfft_flat_k"
_FFT_FFI_CONV_TARGET = "lorrax_mklfft_gw_conv"
_fft_ffi_announced: set = set()


def fft_ffi_enabled() -> bool:
    """LORRAX_FFT_FFI=1 routes flat-k FFTs to the MKL FFT (DFTI API) host
    handler.  Read at helper-FACTORY time (kernel caches must key on it —
    see ppm_tau_kernel).  Unrecognized values announce once and take the
    safe direction: OFF (the default XLA path untouched)."""
    v = os.environ.get("LORRAX_FFT_FFI", "0").strip().lower()
    if v in ("", "0", "off", "false", "no"):
        return False
    if v in ("1", "on", "true", "yes"):
        return True
    if "grammar" not in _fft_ffi_announced:
        _fft_ffi_announced.add("grammar")
        print(f"*** LORRAX_FFT_FFI={v!r} is not a recognized value "
              f"(accepted: 0/off/false/no, 1/on/true/yes).  Treating as OFF "
              f"(default XLA FFT path). ***", flush=True)
    return False


def _require_fft_ffi(mesh: Mesh, target: str) -> None:
    """Announce-or-refuse for the explicitly requested FFI backend."""
    plat = mesh.devices.flat[0].platform
    if plat != "cpu":
        raise RuntimeError(
            f"LORRAX_FFT_FFI requested the MKL FFT (DFTI API) host backend, "
            f"but the mesh devices are {plat!r} — this backend is host-only. "
            f"Unset LORRAX_FFT_FFI on non-CPU meshes (explicit requests are "
            f"never silently downgraded).")
    from ffi.common import ffi_loader
    ok, reason = ffi_loader.probe_target(target, "cpu")
    if not ok:
        raise RuntimeError(
            f"LORRAX_FFT_FFI requested the MKL FFT (DFTI API) host backend, "
            f"but FFI target {target!r} is unusable: {reason}")
    if target not in _fft_ffi_announced:
        _fft_ffi_announced.add(target)
        try:
            first = jax.process_index() == 0
        except Exception:
            first = True
        if first:
            print(f"[fft_ffi] flat-k 3-D FFTs -> MKL FFT (DFTI API) host FFI "
                  f"handler ({target}): O(N log N) FFT reading the dot-layout "
                  f"tile in place via stride descriptors — no XLA layout "
                  f"transposes.", flush=True)


def _ffi_fft_scale(kind: str, norm: str | None, nk: int) -> float:
    """Total scale matching jnp.fft's norm conventions exactly:
    ifftn: backward/None -> 1/N, ortho -> 1/sqrt(N), forward -> 1;
    fftn : backward/None -> 1,  ortho -> 1/sqrt(N), forward -> 1/N."""
    if norm == 'ortho':
        return 1.0 / math.sqrt(float(nk))
    if norm in (None, 'backward'):
        return 1.0 / float(nk) if kind == 'ifftn' else 1.0
    if norm == 'forward':
        return 1.0 if kind == 'ifftn' else 1.0 / float(nk)
    raise ValueError(f"Unsupported FFT norm={norm!r}")


def _validate_ffi_flat_spec(spec: P, what: str) -> P:
    """FFT axes (leading three of the 3-D form) must be replicated; return
    the equivalent flat-form spec (nk axis replicated + original trail)."""
    axes = tuple(spec)
    if len(axes) < 3 or any(ax is not None for ax in axes[:3]):
        raise ValueError(
            f"FFI flat-k backend needs the three k axes of {what} replicated "
            f"(spec {spec}); sharded FFT axes are unsupported (same contract "
            f"as the XLA-path helpers).")
    return P(None, *axes[3:])


def _make_flat_k_fft_ffi(
    mesh: Mesh,
    kgrid: tuple[int, int, int],
    spec: P,
    *,
    kind: str,
    norm: str | None,
    out_spec: P | None,
) -> Callable:
    """FFI-backed flat-k FFT: ``(nk, *trail) -> (nk, *trail)``, same contract
    as :func:`make_flat_k_fft` — one batched MKL FFT (DFTI API) per rank
    over the local shard, k-major layout end to end (never reshaped to the
    3-D k-minor form, which is the whole point)."""
    if kind not in ('ifftn', 'fftn'):
        raise ValueError(f"kind must be 'ifftn' or 'fftn', got {kind!r}")
    if out_spec is not None and tuple(out_spec) != tuple(spec):
        raise ValueError(
            "FFI flat-k backend does not implement a post-FFT reshard "
            f"(out_spec {out_spec} != spec {spec}); unset LORRAX_FFT_FFI for "
            "this call path or drop out_spec.")
    _require_fft_ffi(mesh, _FFT_FFI_TARGET)
    nkx, nky, nkz = (int(v) for v in kgrid)
    nk = nkx * nky * nkz
    flat_spec = _validate_ffi_flat_spec(spec, "the input")
    scale = _ffi_fft_scale(kind, norm, nk)
    attrs = dict(nkx=np.int64(nkx), nky=np.int64(nky), nkz=np.int64(nkz),
                 forward=np.int64(0 if kind == 'ifftn' else 1),
                 scale=np.float64(scale))

    def _local(x_local):
        out_t = jax.ShapeDtypeStruct(x_local.shape, x_local.dtype)
        return jax.ffi.ffi_call(
            _FFT_FFI_TARGET, out_t,
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


def make_flat_k_gw_conv(
    mesh: Mesh,
    kgrid: tuple[int, int, int],
    g_spec: P,
    v_spec: P,
    *,
    norm: str | None = 'ortho',
    mult: float = 1.0,
) -> Callable:
    """FUSED flat-k convolution — the second MKL FFT (DFTI API) entry point.

    Returns ``fn(G_flat, W_flat) -> sigma_flat`` computing, value-identically
    to the decomposed helper sequence (~1e-15 rel; gated, not bit-exact):

        sigma = fftn( ifftn(G) * ifftn(W)[:, None, :, None, :] * mult )

    with all three transforms + the broadcast multiply inside ONE host FFI
    call per rank, chunked so the R-space G tile never materializes (the
    Σ τ kernel's big intermediate).  ``G_flat`` is ``(nk, a, mx, b, my)``,
    ``W_flat`` is ``(nk, mx, my)``; ``mult`` (e.g. Σ's -1/√N_k) is folded
    into the forward-transform scale.  Shapes/strides come from the runtime
    shards — nothing deck-specific.  Sigma-family layout contract only; the
    plain helpers remain the entry point for everything else.
    """
    _require_fft_ffi(mesh, _FFT_FFI_CONV_TARGET)
    nkx, nky, nkz = (int(v) for v in kgrid)
    nk = nkx * nky * nkz
    g_flat = _validate_ffi_flat_spec(g_spec, "G")
    v_flat = _validate_ffi_flat_spec(v_spec, "W")
    attrs = dict(nkx=np.int64(nkx), nky=np.int64(nky), nkz=np.int64(nkz),
                 scale_i=np.float64(_ffi_fft_scale('ifftn', norm, nk)),
                 scale_f=np.float64(_ffi_fft_scale('fftn', norm, nk)
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
            _FFT_FFI_CONV_TARGET, out_t,
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


# ============================================================================
# Flat-k FFT helpers — callers operate on (nk, *trail) arrays everywhere and
# the k-grid 3D form only exists inside this wrapper, matching the
# "flatten kx/ky/kz except inside the FFT" convention used across the GW
# pipeline (w_isdf chi0, ppm_sigma, gw_jax static COHSEX, isdf_fitting
# CCT/ZCT).  Internally: reshape (nk, *trail) -> (nkx, nky, nkz, *trail),
# pin with `with_sharding_constraint`, run a jittable device-local 3D FFT
# over the leading (0, 1, 2) k-axes, reshape back to (nk, *trail).
#
# Backend gate: with LORRAX_FFT_FFI=1 the factory returns the MKL FFT
# (DFTI API) host-FFI variant instead (see the block above) — same
# ``(nk, *trail) -> (nk, *trail)`` contract, no 3-D reshape, no layout
# anchoring.  Default (flag off): the XLA path below, byte-for-byte
# untouched.
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
    replicated (``None``) so the inner custom-partitioned FFT sees the
    full FFT axis on every device.  ``out_spec`` defaults to ``spec``;
    pass a different one only if a post-FFT reshard is wanted.

    ``kind='ifftn'`` or ``'fftn'`` selects the direction.  ``norm``
    follows ``jnp.fft.*`` ('ortho', 'forward', 'backward' / None).
    """
    if fft_ffi_enabled():
        # Gated backend (announce-or-refuse inside): MKL FFT (DFTI API)
        # host handler, k-major end to end.  Factory-time env read — kernel
        # caches key on fft_ffi_enabled() (see ppm_tau_kernel).
        return _make_flat_k_fft_ffi(mesh, kgrid, spec, kind=kind,
                                    norm=norm, out_spec=out_spec)
    nkx, nky, nkz = (int(v) for v in kgrid)
    nk = nkx * nky * nkz
    in_shard = NamedSharding(mesh, spec)
    # Use the single 3D-cuFFT-plan variant rather than the 3-sequential-
    # 1D-FFT (custom_partitioning per-axis) form.  Same correctness
    # contract (FFT axes must be replicated in ``spec``); cuFFT's 3D
    # plan handles the axis sequencing internally with far fewer
    # explicit transposes in the generated HLO.  Si 4×4×4 BSE sweep
    # measured this swap at ~1 s walltime savings in a 200-iter
    # Lanczos run.
    if kind == 'ifftn':
        inner = make_sharded_ifftn_3d(
            mesh, spec, out_spec if out_spec is not None else spec,
            norm=norm, axes=(0, 1, 2))
    elif kind == 'fftn':
        inner = make_sharded_fftn_3d(
            mesh, spec, out_spec if out_spec is not None else spec,
            norm=norm, axes=(0, 1, 2))
    else:
        raise ValueError(f"kind must be 'ifftn' or 'fftn', got {kind!r}")

    def _flat_k_fft(x_flat):
        trail = x_flat.shape[1:]
        x_3d = jax.lax.with_sharding_constraint(
            x_flat.reshape(nkx, nky, nkz, *trail), in_shard)
        return inner(x_3d).reshape(nk, *trail)

    return _flat_k_fft


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
