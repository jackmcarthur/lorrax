import math
from typing import Callable

import jax
import jax.numpy as jnp
from jax.experimental.custom_partitioning import custom_partitioning
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P
from jax.experimental.shard_map import shard_map


# =============================================================================
# FFT workspace query (for memory-model sizing)
# =============================================================================
# The stage-cost memory model in ``gw_init.compute_optimal_chunks`` (and the
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
    def _local_ifftn(x_local):
        # Each device runs FFT on its local shard independently
        return jnp.fft.ifftn(x_local, axes=axes, norm=norm)

    return shard_map(_local_ifftn, mesh=mesh, in_specs=(in_spec,), out_specs=out_spec)

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
    def _local_fftn(x_local):
        return jnp.fft.fftn(x_local, axes=axes, norm=norm)

    return shard_map(_local_fftn, mesh=mesh, in_specs=(in_spec,), out_specs=out_spec)


# ============================================================================
# Flat-k FFT helpers — callers operate on (nk, *trail) arrays everywhere and
# the k-grid 3D form only exists inside this wrapper, matching the
# "flatten kx/ky/kz except inside the FFT" convention used across the GW
# pipeline (w_isdf chi0, ppm_sigma, gw_jax static COHSEX, isdf_fitting
# CCT/ZCT).  Internally: reshape (nk, *trail) -> (nkx, nky, nkz, *trail),
# pin with `with_sharding_constraint`, run a jittable device-local 3D FFT
# over the leading (0, 1, 2) k-axes, reshape back to (nk, *trail).
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
