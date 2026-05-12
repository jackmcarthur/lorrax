import math
from typing import Callable

import jax
import jax.numpy as jnp
import numpy as np
from jax.experimental.shard_map import shard_map
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P


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
# the exact local ``jnp.fft.fftn`` XLA would emit, read its
# ``memory_analysis()``, and cache the result — pure JAX, works wherever JAX
# works.
#
# This function is called statically at chooser time (never in a hot loop), so
# the ~1-2 s per-shape compile cost is amortised across the full run.  Two
# canonical uses in the pipeline:
#   * Wavefunction-box FFT:   shape = fft_grid (nx, ny, nz), batched by
#                             nk × bpd × ns (in-loop ψ_G → ψ_r).
#   * k-grid FFT (ZCT / CCT): shape = kgrid (nkx, nky, nkz), batched by
#                             μ × cr or μ × μ.

_fft_workspace_cache: dict = {}


def _prod(shape: tuple[int, ...]) -> int:
    return math.prod(int(s) for s in shape)


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
                "Local FFT helpers require every transformed axis to be replicated. "
                f"Axis {ax} is sharded in in_spec={in_spec}, out_spec={out_spec}."
            )
    return fft_axes


def _validate_fft_batch_chunks(fft_batch_chunks: int) -> int:
    chunks = int(fft_batch_chunks)
    if chunks < 1:
        raise ValueError(f"fft_batch_chunks must be >= 1, got {fft_batch_chunks}.")
    return chunks


def _apply_local_fft_chunked(
    x: jax.Array,
    *,
    fft_op,
    fft_axes: tuple[int, ...],
    norm: str | None,
    fft_batch_chunks: int,
) -> jax.Array:
    batch_axes = tuple(ax for ax in range(x.ndim) if ax not in fft_axes)
    perm = batch_axes + fft_axes
    inv_perm = tuple(int(i) for i in np.argsort(np.asarray(perm)))
    x_perm = jnp.transpose(x, perm)

    batch_shape = tuple(int(x.shape[ax]) for ax in batch_axes)
    fft_shape = tuple(int(x.shape[ax]) for ax in fft_axes)
    batch_size = _prod(batch_shape) if batch_shape else 1
    chunk_count = _validate_fft_batch_chunks(fft_batch_chunks)
    chunk_batch = max(1, -(-batch_size // chunk_count))
    padded_batch = chunk_batch * chunk_count

    x_flat = x_perm.reshape(batch_size, *fft_shape)
    if padded_batch != batch_size:
        pad_width = ((0, padded_batch - batch_size),) + ((0, 0),) * len(fft_shape)
        x_flat = jnp.pad(x_flat, pad_width)
    x_chunks = x_flat.reshape(chunk_count, chunk_batch, *fft_shape)
    chunk_fft_axes = tuple(range(1, 1 + len(fft_axes)))

    def _fft_chunk(_, chunk):
        return None, fft_op(chunk, axes=chunk_fft_axes, norm=norm)

    _, y_chunks = jax.lax.scan(_fft_chunk, None, x_chunks, unroll=1)
    y_flat = y_chunks.reshape(padded_batch, *fft_shape)[:batch_size]
    y_perm = y_flat.reshape(*batch_shape, *fft_shape) if batch_shape else y_flat.reshape(*fft_shape)
    return jnp.transpose(y_perm, inv_perm)


def apply_local_fft(
    x: jax.Array,
    *,
    fft_kind: str,
    axes: tuple[int, ...],
    norm: str | None = None,
    fft_batch_chunks: int = 1,
) -> jax.Array:
    """Run a local FFT on ``x`` with optional chunking over non-FFT axes."""
    if fft_kind not in ("ifftn", "fftn"):
        raise ValueError(f"Unsupported fft_kind={fft_kind!r}")
    fft_axes = _normalize_local_fft_axes(x.ndim, axes)
    fft_op = jnp.fft.ifftn if fft_kind == "ifftn" else jnp.fft.fftn
    chunk_count = _validate_fft_batch_chunks(fft_batch_chunks)
    if chunk_count == 1:
        return fft_op(x, axes=fft_axes, norm=norm)
    return _apply_local_fft_chunked(
        x,
        fft_op=fft_op,
        fft_axes=fft_axes,
        norm=norm,
        fft_batch_chunks=chunk_count,
    )


def _make_local_fft_impl(
    *,
    fft_kind: str,
    norm: str | None,
    fft_axes: tuple[int, ...],
    fft_batch_chunks: int,
):
    """Return the device-local FFT implementation used by every helper."""
    chunk_count = _validate_fft_batch_chunks(fft_batch_chunks)
    return lambda x: apply_local_fft(
        x,
        fft_kind=fft_kind,
        axes=fft_axes,
        norm=norm,
        fft_batch_chunks=chunk_count,
    )


def _make_sharded_fft(
    mesh: Mesh,
    in_spec: P,
    out_spec: P,
    *,
    fft_kind: str,
    norm: str | None,
    axes: tuple[int, ...],
    fft_batch_chunks: int,
):
    """shard_map local FFT preserving sharding on replicated FFT axes."""
    fft_axes = _validate_local_fft_specs(in_spec, out_spec, axes)
    local_fft = _make_local_fft_impl(
        fft_kind=fft_kind,
        norm=norm,
        fft_axes=fft_axes,
        fft_batch_chunks=fft_batch_chunks,
    )
    return shard_map(local_fft, mesh=mesh, in_specs=(in_spec,), out_specs=out_spec)


def _query_fft_peak_bytes_impl(
    *,
    input_shape: tuple[int, ...],
    fft_axes: tuple[int, ...],
    sharding: NamedSharding,
    dtype,
    fft_batch_chunks: int,
) -> int:
    mesh = sharding.mesh
    chunk_count = _validate_fft_batch_chunks(fft_batch_chunks)
    key = (
        tuple(input_shape),
        tuple(fft_axes),
        str(sharding.spec),
        jnp.dtype(dtype).str,
        tuple(mesh.axis_names),
        tuple(int(mesh.shape[a]) for a in mesh.axis_names),
        chunk_count,
    )
    hit = _fft_workspace_cache.get(key)
    if hit is not None:
        return hit

    spec = jax.ShapeDtypeStruct(
        tuple(int(s) for s in input_shape), dtype, sharding=sharding)
    local_fftn = _make_sharded_fft(
        mesh,
        sharding.spec,
        sharding.spec,
        fft_kind="fftn",
        norm=None,
        axes=tuple(fft_axes),
        fft_batch_chunks=chunk_count,
    )
    jit_fft = jax.jit(local_fftn, out_shardings=sharding)

    try:
        compiled = jit_fft.lower(spec).compile(
            compiler_options={"xla_gpu_memory_limit_slop_factor": 10000})
    except Exception:
        elem = jnp.dtype(dtype).itemsize
        total_elems = _prod(tuple(int(s) for s in input_shape))
        n_devs = 1
        for a in mesh.axis_names:
            n_devs *= int(mesh.shape[a])
        fallback = 3 * total_elems * elem // max(1, n_devs)
        _fft_workspace_cache[key] = fallback
        return fallback

    m = compiled.memory_analysis()
    total = (
        int(m.temp_size_in_bytes)
        + int(m.argument_size_in_bytes)
        + int(m.output_size_in_bytes)
        - int(m.alias_size_in_bytes)
    )
    _fft_workspace_cache[key] = total
    return total


def query_fft_peak_bytes(
    *,
    input_shape: tuple[int, ...],
    fft_axes: tuple[int, ...],
    sharding: NamedSharding,
    dtype=jnp.complex128,
    fft_batch_chunks: int = 1,
) -> int:
    """AOT-compile the default local FFT path and return per-rank peak bytes."""
    return _query_fft_peak_bytes_impl(
        input_shape=input_shape,
        fft_axes=fft_axes,
        sharding=sharding,
        dtype=dtype,
        fft_batch_chunks=fft_batch_chunks,
    )


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

    for j_mult in range(2, 20):
        J = target_J * j_mult
        if n_rmu % J == 0:
            block_size = n_rmu // J
            return block_size, J

    for b in range(n_rmu, 0, -1):
        if n_rmu % b == 0:
            J = n_rmu // b
            if J % Pr == 0 and J % Pc == 0:
                return b, J

    raise ValueError(
        f"No valid block size for n_rmu={n_rmu} with mesh {Pr}×{Pc}. "
        f"n_rmu should be divisible by lcm({Pr},{Pc})={target_J} or a multiple thereof."
    )


def make_jittable_local_ifftn_3d(
    mesh: Mesh,
    in_spec: P,
    out_spec: P,
    *,
    norm: str | None = None,
    axes: tuple[int, int, int] = (-3, -2, -1),
    fft_batch_chunks: int = 1,
):
    """Legacy name kept as an alias for the production shard_map helper."""
    return make_sharded_ifftn_3d(
        mesh, in_spec, out_spec, norm=norm, axes=axes,
        fft_batch_chunks=fft_batch_chunks)


def make_jittable_local_fftn_3d(
    mesh: Mesh,
    in_spec: P,
    out_spec: P,
    *,
    norm: str | None = None,
    axes: tuple[int, int, int] = (-3, -2, -1),
    fft_batch_chunks: int = 1,
):
    """Legacy name kept as an alias for the production shard_map helper."""
    return make_sharded_fftn_3d(
        mesh, in_spec, out_spec, norm=norm, axes=axes,
        fft_batch_chunks=fft_batch_chunks)


def make_sharded_ifftn_3d(
    mesh: Mesh,
    in_spec: P,
    out_spec: P,
    *,
    norm: str | None = None,
    axes: tuple[int, int, int] = (-3, -2, -1),
    fft_batch_chunks: int = 1,
):
    """
    Uses shard_map to run IFFT independently on each device's local data.
    The FFT axes must not be sharded; only batch dims may be sharded.
    """
    return _make_sharded_fft(
        mesh,
        in_spec,
        out_spec,
        fft_kind="ifftn",
        norm=norm,
        axes=axes,
        fft_batch_chunks=fft_batch_chunks,
    )


def make_sharded_fftn_3d(
    mesh: Mesh,
    in_spec: P,
    out_spec: P,
    *,
    norm: str | None = None,
    axes: tuple[int, int, int] = (-3, -2, -1),
    fft_batch_chunks: int = 1,
):
    """Forward shard_map local FFT."""
    return _make_sharded_fft(
        mesh,
        in_spec,
        out_spec,
        fft_kind="fftn",
        norm=norm,
        axes=axes,
        fft_batch_chunks=fft_batch_chunks,
    )


# ============================================================================
# Flat-k FFT helpers — callers operate on (nk, *trail) arrays everywhere and
# the k-grid 3D form only exists inside this wrapper, matching the
# "flatten kx/ky/kz except inside the FFT" convention used across the GW
# pipeline (w_isdf chi0, ppm_sigma, gw_jax static COHSEX, isdf_fitting
# CCT/ZCT).  Internally: reshape (nk, *trail) -> (nkx, nky, nkz, *trail),
# pin with `with_sharding_constraint`, run a device-local 3D FFT over the
# leading (0, 1, 2) k-axes, reshape back to (nk, *trail).
# ============================================================================


def make_flat_k_fft(
    mesh: Mesh,
    kgrid: tuple[int, int, int],
    spec: P,
    *,
    kind: str,
    norm: str | None = "ortho",
    out_spec: P | None = None,
    fft_batch_chunks: int = 1,
) -> Callable:
    """Return a flat-k FFT: ``(nk, *trail) -> (nk, *trail)``."""
    nkx, nky, nkz = (int(v) for v in kgrid)
    nk = nkx * nky * nkz
    in_shard = NamedSharding(mesh, spec)

    if kind == "ifftn":
        inner = make_sharded_ifftn_3d(
            mesh, spec, out_spec if out_spec is not None else spec,
            norm=norm, axes=(0, 1, 2),
            fft_batch_chunks=fft_batch_chunks)
    elif kind == "fftn":
        inner = make_sharded_fftn_3d(
            mesh, spec, out_spec if out_spec is not None else spec,
            norm=norm, axes=(0, 1, 2),
            fft_batch_chunks=fft_batch_chunks)
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
    norm: str | None = "ortho",
    out_spec: P | None = None,
    fft_batch_chunks: int = 1,
) -> Callable:
    """Flat-k IFFT ``(nk, *trail) -> (nk, *trail)``.  See :func:`make_flat_k_fft`."""
    return make_flat_k_fft(mesh, kgrid, spec, kind="ifftn",
                           norm=norm, out_spec=out_spec,
                           fft_batch_chunks=fft_batch_chunks)


def make_flat_k_fftn(
    mesh: Mesh,
    kgrid: tuple[int, int, int],
    spec: P,
    *,
    norm: str | None = "ortho",
    out_spec: P | None = None,
    fft_batch_chunks: int = 1,
) -> Callable:
    """Flat-k FFT ``(nk, *trail) -> (nk, *trail)``.  See :func:`make_flat_k_fft`."""
    return make_flat_k_fft(mesh, kgrid, spec, kind="fftn",
                           norm=norm, out_spec=out_spec,
                           fft_batch_chunks=fft_batch_chunks)
