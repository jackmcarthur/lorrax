"""Transforms from G-flat ψ to FFT-box / r-space / centroid / r-chunk.

Composes with :class:`file_io.wfn_loader.WfnLoader`.  The loader returns
``psi`` in G-flat layout ``(n_k, nb_padded, nspinor, ngkmax)`` c128 and a
``g_index`` from :meth:`WfnLoader.box_index`; this module turns either
pair into the downstream product the consumer actually needs (FFT box,
r-space box, ψ at centroid indices, ψ on a flat-r slab).

Why split this off
------------------
g_flat is ~6-11% of the FFT-box size; band-chunked GW loops that only
need ψ at centroids should never materialise the full FFT box.  Keeping
these as standalone composable functions (rather than methods on the
loader) lets a fused-NUFFT variant land later without changing the
loader API.

Sharding contract
-----------------
Every transform **preserves the band-axis sharding** of its input ``psi``.
The default sharding from ``WfnLoader.load`` is
``P(None, ('x','y'), None, None)`` (band sharded across the 2-D mesh);
outputs add inner axes as ``P(None, ('x','y'), None, None, None, None)``
(FFT-box) or ``P(None, ('x','y'), None, None)`` (r-chunk / r-mu).  No
cross-rank communication is required by any transform.

Replicated ``psi`` (single-rank pytest, or callers passing
``sharding=None`` to the loader) goes through a non-shard_map jit fast
path so the transforms work on a laptop without a mesh.

Public API
----------
* :func:`to_box`   — G-flat → FFT-box  ``(n_k, nb, ns, nx, ny, nz)``
* :func:`to_rbox`  — G-flat → r-space FFT-box  (= IFFT(to_box))
* :func:`to_rmu`   — G-flat → ψ at centroid indices ``(n_k, nb, ns, n_rmu)``
* :func:`to_rchunk` — G-flat → ψ on flat-r slab ``(n_k, nb, ns, r_len)``

All four use the same gather kernel internally; the variants differ only
in what happens after the IFFT.
"""
from __future__ import annotations

from functools import partial
from typing import Sequence

import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import NamedSharding, PartitionSpec as P

from .fft_helpers import apply_local_fft

__all__ = [
    "to_box",
    "to_rbox",
    "to_rmu",
    "to_rchunk",
    "apply_bloch_phase",
    "apply_bloch_phase_flat_points",
    "apply_bloch_phase_flat_rchunk",
    "extract_flat_rchunk",
    "embed_flat_rchunk",
]


# ---------------------------------------------------------------------------
# Kernel cache
# ---------------------------------------------------------------------------

_BOX_KERNEL_CACHE: dict = {}
_RBOX_KERNEL_CACHE: dict = {}
_RMU_KERNEL_CACHE: dict = {}
_RCHUNK_KERNEL_CACHE: dict = {}


# ---------------------------------------------------------------------------
# Shared kernel: G-flat ψ + g_index → FFT-box ψ (zero-sentinel gather)
# ---------------------------------------------------------------------------
#
# Algorithm (matches ``common/gvec_fft_box.make_fft_box_kernel`` but on
# WfnLoader's c128 layout).  For each FFT-box cell (nx, ny, nz),
# ``g_index[k, nx, ny, nz]`` gives the position along the G-axis of psi
# to gather from; positions equal to ``ngkmax`` map to a synthetic
# zero slot appended on the G-axis before the gather.  One ``take``
# call fills the whole box; no per-k loop, no scatter, no per-cell
# masking.

def _box_kernel(psi: jax.Array, g_index: jax.Array, *, ngkmax: int) -> jax.Array:
    """psi: (n_k, nb, ns, ngkmax) c128 — band-sharded acceptable.
    g_index: (n_k, nx, ny, nz) int32 replicated.
    Returns (n_k, nb, ns, nx, ny, nz) c128 with band sharding preserved.

    Pure jax — no shard_map.  Sharding propagates by XLA's normal rules:
    the gather is over the G-axis (axis 3 of psi after the reshape +
    transpose dance), no cross-rank op required.
    """
    n_k, nb, ns, _ = psi.shape
    k_stride = ngkmax + 1
    # Append a zero slot on the G-axis so sentinel index `ngkmax`
    # gathers zero.
    zero = jnp.zeros((n_k, nb, ns, 1), dtype=psi.dtype)
    psi_padded = jnp.concatenate([psi, zero], axis=-1)            # (..., ngkmax+1)
    # Move (nb, ns) to the front so the gather indexes the (k, g) plane.
    psi_t = jnp.transpose(psi_padded, (1, 2, 0, 3))               # (nb, ns, n_k, ngkmax+1)
    psi_flat = psi_t.reshape(nb, ns, n_k * k_stride)
    # Per-cell flat index combining k and g.
    flat_index = (
        jnp.arange(n_k, dtype=jnp.int32)[:, None, None, None] * k_stride
        + g_index)                                                # (n_k, nx, ny, nz)
    # ``mode='clip'`` skips the OOB ``_where`` mask jnp.take inserts under
    # the default ``mode='fill'``.  By construction ``flat_index`` is in
    # range (psi was padded with a zero slot at index ``ngkmax``), so the
    # clip is a no-op on the values — and avoids the per-shape ``_where``
    # retraces (8 cache misses in MoS2 3×3 profile before this change).
    gathered = jnp.take(psi_flat, flat_index, axis=2, mode='clip')  # (nb, ns, n_k, nx, ny, nz)
    return jnp.transpose(gathered, (2, 0, 1, 3, 4, 5))


# ---------------------------------------------------------------------------
# Output sharding propagation
# ---------------------------------------------------------------------------

def _band_spec_of(psi: jax.Array) -> tuple | None:
    """Return the band-axis entry of psi's PartitionSpec, or None if not
    sharded.  Used to propagate the band shard to higher-dim outputs."""
    try:
        sharding = psi.sharding
    except AttributeError:
        return None
    spec = list(getattr(sharding, "spec", ())) if sharding is not None else []
    if len(spec) < 2:
        return None
    return spec[1]


def _output_sharding(psi: jax.Array, n_extra_axes: int) -> NamedSharding | None:
    """Build a NamedSharding for an output with ``n_extra_axes`` axes
    inserted after the ``(n_k, nb, ns)`` prefix — all replicated — and
    the band axis preserved."""
    try:
        mesh = psi.sharding.mesh
    except AttributeError:
        return None
    band_spec = _band_spec_of(psi)
    if band_spec is None and isinstance(psi.sharding, NamedSharding):
        # Replicated input on a real mesh — replicate output too.
        return NamedSharding(psi.sharding.mesh, P(*[None] * (3 + n_extra_axes)))
    spec_parts = [None, band_spec, None] + [None] * n_extra_axes
    return NamedSharding(mesh, P(*spec_parts))


def _maybe_constrain(arr: jax.Array, sharding: NamedSharding | None) -> jax.Array:
    if sharding is None:
        return arr
    return jax.lax.with_sharding_constraint(arr, sharding)


def _flat_points_to_coords(
    r_flat: jax.Array,
    fft_grid: tuple[int, int, int],
) -> tuple[jax.Array, jax.Array, jax.Array]:
    nx, ny, nz = (int(s) for s in fft_grid)
    del nx
    yz = ny * nz
    rx = r_flat // yz
    rem = r_flat % yz
    ry = rem // nz
    rz = rem % nz
    return rx, ry, rz


def apply_bloch_phase_flat_points(
    values: jax.Array,
    kvecs_frac: jax.Array,
    r_flat: jax.Array,
    fft_grid: tuple[int, int, int],
    *,
    sign: int = 1,
) -> jax.Array:
    """Apply ``exp(sign · 2πi k·r)`` on a flat-r point list only."""
    nx, ny, nz = (int(s) for s in fft_grid)
    r_flat = jnp.asarray(r_flat, dtype=jnp.int32)
    rx, ry, rz = _flat_points_to_coords(r_flat, fft_grid)
    phase_arg = (
        kvecs_frac[:, 0:1] * (rx[None, :] / nx)
        + kvecs_frac[:, 1:2] * (ry[None, :] / ny)
        + kvecs_frac[:, 2:3] * (rz[None, :] / nz)
    )
    phase = jnp.exp(jnp.complex128(int(sign) * 2j * jnp.pi) * phase_arg)
    n_mid = values.ndim - 2
    phase = phase.reshape(phase.shape[0], *((1,) * n_mid), phase.shape[1])
    return values * phase


def apply_bloch_phase_flat_rchunk(
    values: jax.Array,
    kvecs_frac: jax.Array,
    fft_grid: tuple[int, int, int],
    r0,
    *,
    sign: int = 1,
) -> jax.Array:
    """Apply the Bloch phase on a contiguous flat-r slab only."""
    r_len = int(values.shape[-1])
    r_flat = jnp.arange(r_len, dtype=jnp.int32) + jnp.asarray(r0, dtype=jnp.int32)
    return apply_bloch_phase_flat_points(
        values, kvecs_frac, r_flat, fft_grid, sign=sign)


def extract_flat_rchunk(box: jax.Array, r0, r_len: int) -> jax.Array:
    """Flatten the trailing FFT-box axes and slice a contiguous r slab."""
    n_rtot = int(np.prod(np.asarray(box.shape[-3:], dtype=np.int64)))
    box_flat = box.reshape(*box.shape[:-3], n_rtot)
    return jax.lax.dynamic_slice_in_dim(box_flat, r0, int(r_len), axis=-1)


def embed_flat_rchunk(chunk: jax.Array, fft_grid: tuple[int, int, int], r0) -> jax.Array:
    """Embed a flat-r slab into a zero-padded FFT box with trailing 3D axes."""
    nx, ny, nz = (int(s) for s in fft_grid)
    n_rtot = nx * ny * nz
    flat = jnp.zeros((*chunk.shape[:-1], n_rtot), dtype=chunk.dtype)
    flat = jax.lax.dynamic_update_slice_in_dim(flat, chunk, r0, axis=-1)
    return flat.reshape(*chunk.shape[:-1], nx, ny, nz)


# ---------------------------------------------------------------------------
# Sharding signature key — used to keep the jit caches small and stable.
# ---------------------------------------------------------------------------

def _sharding_key(psi: jax.Array) -> tuple:
    """Hashable signature of psi's sharding (mesh identity + spec).

    Used as part of the jit-cache key for the public transforms so two
    arrays with identical mesh + PartitionSpec hit the same compiled
    XLA module, while a different mesh forces a fresh compile."""
    sh = getattr(psi, "sharding", None)
    if sh is None:
        return ("no_sharding",)
    mesh_id = id(getattr(sh, "mesh", None))
    spec = tuple(getattr(sh, "spec", ()))
    return (mesh_id, spec)


# ---------------------------------------------------------------------------
# Public transforms — shape-keyed jit caches.  Each public ``to_X`` looks
# up a compiled closure that fuses (gather → optional IFFT → optional
# Bloch-phase → optional slice/gather → sharding constraint) into ONE XLA
# module per (input shape, fft_grid, extra-arg shape, sharding) key.
#
# Mirrors the cache pattern that the old ``get_sharded_wfns_rchunk_slice``
# / ``get_sharded_wfns_centroids`` used (``_rchunk_slice_cache`` /
# ``_centroid_extract_cache`` in ``load_wfns.py``).
# ---------------------------------------------------------------------------

def to_box(
    psi: jax.Array,
    g_index: np.ndarray | jax.Array,
    fft_grid: Sequence[int],
) -> jax.Array:
    """Scatter G-flat ψ into the FFT box.

    Sharding (band axis on ``('x','y')`` or replicated) is preserved.
    ``g_index`` (output of :meth:`WfnLoader.box_index`) uses sentinel
    ``ngkmax`` to flag empty FFT-box cells (zero on gather).
    """
    ngkmax = int(psi.shape[-1])
    fft_grid_t = tuple(int(s) for s in fft_grid)
    out_sharding = _output_sharding(psi, n_extra_axes=3)
    key = (psi.shape, tuple(g_index.shape), ngkmax, fft_grid_t,
           _sharding_key(psi), out_sharding)
    fn = _BOX_KERNEL_CACHE.get(key)
    if fn is None:
        @jax.jit
        def fn(psi_, g_index_):
            out = _box_kernel(psi_, g_index_, ngkmax=ngkmax)
            return _maybe_constrain(out, out_sharding)
        _BOX_KERNEL_CACHE[key] = fn
    g_index_j = jnp.asarray(g_index, dtype=jnp.int32)
    return fn(psi, g_index_j)


def to_rbox(
    psi: jax.Array,
    g_index: np.ndarray | jax.Array,
    fft_grid: Sequence[int],
    *,
    norm: str = "backward",
    kvecs_frac: np.ndarray | jax.Array | None = None,
    fft_batch_chunks: int = 1,
) -> jax.Array:
    """Scatter ψ → FFT box → IFFT to r-space (+ optional Bloch phase).

    ``norm`` is forwarded to :func:`jnp.fft.ifftn` (``'backward'`` =
    ``1/N``; ``'ortho'`` = ``1/√N`` on both directions, used by the
    centroid pivoted-Cholesky path).  ``kvecs_frac`` (n_k, 3) optionally
    applies ``exp(+2πi k·r)`` after the IFFT (set to ``None`` for the
    ``|ψ|²``-only path).  Output materialises the full FFT box; prefer
    :func:`to_rmu`/:func:`to_rchunk` for centroid / slab consumers.
    """
    ngkmax = int(psi.shape[-1])
    fft_grid_t = tuple(int(s) for s in fft_grid)
    kvecs_shape = (None if kvecs_frac is None
                   else tuple(int(s) for s in np.shape(kvecs_frac)))
    out_sharding = _output_sharding(psi, n_extra_axes=3)
    key = (psi.shape, tuple(g_index.shape), ngkmax, fft_grid_t, norm,
           kvecs_shape, int(fft_batch_chunks), _sharding_key(psi), out_sharding)
    fn = _RBOX_KERNEL_CACHE.get(key)
    if fn is None:
        if kvecs_frac is None:
            @jax.jit
            def fn(psi_, g_index_):
                box = _box_kernel(psi_, g_index_, ngkmax=ngkmax)
                out = apply_local_fft(
                    box, fft_kind="ifftn", axes=(-3, -2, -1), norm=norm,
                    fft_batch_chunks=fft_batch_chunks)
                return _maybe_constrain(out, out_sharding)
            _RBOX_KERNEL_CACHE[key] = fn
        else:
            @jax.jit
            def fn(psi_, g_index_, kvecs_):
                box = _box_kernel(psi_, g_index_, ngkmax=ngkmax)
                rb = apply_local_fft(
                    box, fft_kind="ifftn", axes=(-3, -2, -1), norm=norm,
                    fft_batch_chunks=fft_batch_chunks)
                rb = apply_bloch_phase(rb, kvecs_, fft_grid_t)
                return _maybe_constrain(rb, out_sharding)
            _RBOX_KERNEL_CACHE[key] = fn
    g_index_j = jnp.asarray(g_index, dtype=jnp.int32)
    if kvecs_frac is None:
        return fn(psi, g_index_j)
    return fn(psi, g_index_j, jnp.asarray(kvecs_frac, dtype=jnp.float64))


def to_rmu(
    psi: jax.Array,
    g_index: np.ndarray | jax.Array,
    fft_grid: Sequence[int],
    r_mu: np.ndarray | jax.Array,
    *,
    norm: str = "backward",
    kvecs_frac: np.ndarray | jax.Array | None = None,
    fft_batch_chunks: int = 1,
) -> jax.Array:
    """ψ in r-space at the centroid FFT-grid indices ``r_mu``.

    ``r_mu`` is ``(n_rmu, 3)`` int32 (positions in ``[0, fft_grid[a])``);
    other args as :func:`to_rbox`.  Output ``(n_k, nb, nspinor, n_rmu)``
    with band-axis sharding preserved.
    """
    ngkmax = int(psi.shape[-1])
    fft_grid_t = tuple(int(s) for s in fft_grid)
    n_rmu = int(np.shape(r_mu)[0])
    r_mu_flat = (
        np.asarray(r_mu)[..., 0] * fft_grid_t[1] * fft_grid_t[2]
        + np.asarray(r_mu)[..., 1] * fft_grid_t[2]
        + np.asarray(r_mu)[..., 2]
    )
    kvecs_shape = (None if kvecs_frac is None
                   else tuple(int(s) for s in np.shape(kvecs_frac)))
    out_sharding = _output_sharding(psi, n_extra_axes=1)
    key = (psi.shape, tuple(g_index.shape), ngkmax, fft_grid_t, n_rmu,
           norm, kvecs_shape, int(fft_batch_chunks), _sharding_key(psi),
           out_sharding)
    fn = _RMU_KERNEL_CACHE.get(key)
    if fn is None:
        if kvecs_frac is None:
            @jax.jit
            def fn(psi_, g_index_, r_mu_):
                box = _box_kernel(psi_, g_index_, ngkmax=ngkmax)
                rb = apply_local_fft(
                    box, fft_kind="ifftn", axes=(-3, -2, -1), norm=norm,
                    fft_batch_chunks=fft_batch_chunks)
                out = rb[:, :, :, r_mu_[:, 0], r_mu_[:, 1], r_mu_[:, 2]]
                return _maybe_constrain(out, out_sharding)
            _RMU_KERNEL_CACHE[key] = fn
        else:
            @jax.jit
            def fn(psi_, g_index_, r_mu_, r_mu_flat_, kvecs_):
                box = _box_kernel(psi_, g_index_, ngkmax=ngkmax)
                rb = apply_local_fft(
                    box, fft_kind="ifftn", axes=(-3, -2, -1), norm=norm,
                    fft_batch_chunks=fft_batch_chunks)
                out = rb[:, :, :, r_mu_[:, 0], r_mu_[:, 1], r_mu_[:, 2]]
                out = apply_bloch_phase_flat_points(
                    out, kvecs_, r_mu_flat_, fft_grid_t)
                return _maybe_constrain(out, out_sharding)
            _RMU_KERNEL_CACHE[key] = fn
    g_index_j = jnp.asarray(g_index, dtype=jnp.int32)
    r_mu_j = jnp.asarray(r_mu, dtype=jnp.int32)
    if kvecs_frac is None:
        return fn(psi, g_index_j, r_mu_j)
    return fn(psi, g_index_j, r_mu_j, jnp.asarray(r_mu_flat, dtype=jnp.int32),
              jnp.asarray(kvecs_frac, dtype=jnp.float64))


def to_rchunk(
    psi: jax.Array,
    g_index: np.ndarray | jax.Array,
    fft_grid: Sequence[int],
    r0,
    r_len: int,
    *,
    norm: str = "backward",
    kvecs_frac: np.ndarray | jax.Array | None = None,
    fft_batch_chunks: int = 1,
) -> jax.Array:
    """ψ in r-space on a contiguous flat-r slab ``[r0, r0 + r_len)``.

    Flat-r convention: ``r_flat = rx * ny * nz + ry * nz + rz``.  ``r0``
    may be a Python int (bounds-checked) or a traced scalar (caller's
    responsibility).
    """
    ngkmax = int(psi.shape[-1])
    fft_grid_t = tuple(int(s) for s in fft_grid)
    nx, ny, nz = fft_grid_t
    n_rtot = nx * ny * nz
    r_len_i = int(r_len)
    if isinstance(r0, (int, np.integer)):
        if r0 < 0 or int(r0) + r_len_i > n_rtot:
            raise ValueError(
                f"to_rchunk: [{int(r0)}, {int(r0) + r_len_i}) out of [0, {n_rtot})")
    kvecs_shape = (None if kvecs_frac is None
                   else tuple(int(s) for s in np.shape(kvecs_frac)))
    out_sharding = _output_sharding(psi, n_extra_axes=1)
    key = (psi.shape, tuple(g_index.shape), ngkmax, fft_grid_t, r_len_i,
           norm, kvecs_shape, int(fft_batch_chunks), _sharding_key(psi),
           out_sharding)
    fn = _RCHUNK_KERNEL_CACHE.get(key)
    if fn is None:
        if kvecs_frac is None:
            @jax.jit
            def fn(psi_, g_index_, r0_):
                box = _box_kernel(psi_, g_index_, ngkmax=ngkmax)
                rb = apply_local_fft(
                    box, fft_kind="ifftn", axes=(-3, -2, -1), norm=norm,
                    fft_batch_chunks=fft_batch_chunks)
                out = extract_flat_rchunk(rb, r0_, r_len_i)
                return _maybe_constrain(out, out_sharding)
            _RCHUNK_KERNEL_CACHE[key] = fn
        else:
            @jax.jit
            def fn(psi_, g_index_, r0_, kvecs_):
                box = _box_kernel(psi_, g_index_, ngkmax=ngkmax)
                rb = apply_local_fft(
                    box, fft_kind="ifftn", axes=(-3, -2, -1), norm=norm,
                    fft_batch_chunks=fft_batch_chunks)
                out = extract_flat_rchunk(rb, r0_, r_len_i)
                out = apply_bloch_phase_flat_rchunk(out, kvecs_, fft_grid_t, r0_)
                return _maybe_constrain(out, out_sharding)
            _RCHUNK_KERNEL_CACHE[key] = fn
    g_index_j = jnp.asarray(g_index, dtype=jnp.int32)
    r0_arg = (jnp.int32(int(r0)) if isinstance(r0, (int, np.integer)) else r0)
    if kvecs_frac is None:
        return fn(psi, g_index_j, r0_arg)
    return fn(psi, g_index_j, r0_arg,
              jnp.asarray(kvecs_frac, dtype=jnp.float64))


# ---------------------------------------------------------------------------
# Bloch phase ``exp(σ · 2πi k·r)`` applied separably in x / y / z.
# ---------------------------------------------------------------------------
#
# ``exp(σ · 2πi k·r) = exp(σ·2πi kx fx) · exp(σ·2πi ky fy) · exp(σ·2πi kz fz)``.
# The three 1D factors are kept apart and broadcast-multiplied in
# sequence against the input box.  XLA fuses the three pointwise
# multiplies into a single pass.  Scratch memory: ``n_k · (nx + ny + nz)``
# complex128 per axis — three orders of magnitude smaller than the
# full ``n_k · nx · ny · nz`` 4D phase that an explicit-product
# implementation would materialise.
#
# Single source of truth: this is the ONLY place in LORRAX where the
# Bloch-phase formula lives.  Used by:
#   * ψ-r box pipeline (post-IFFT, sign='+'):    to_rbox / to_rmu / to_rchunk
#     for ``ψ_nk(r) = exp(+2πi k·r) · u_nk(r)``
#   * ζ-r → G FFT (pre-FFT, sign='-'):           v_q_tile._zeta_disk_to_G
#     and file_io.zeta_reader._do_disk_to_G for ``z_q,μ(r) = exp(-2πi q·r) ζ_q,μ(r)``
#     before scattering onto the (q + G) sphere.

def apply_bloch_phase(
    box: jax.Array,
    kvecs_frac: jax.Array,
    fft_grid: tuple[int, int, int],
    *,
    sign: int = 1,
) -> jax.Array:
    """box × exp(sign · 2πi k·r) applied as three separable 1D multiplies.

    ``box``: trailing shape ``(..., nx, ny, nz)`` c128 (sharding preserved).
        The leading axis must be the k-axis whose length matches
        ``kvecs_frac.shape[0]``.  Any number of intermediate broadcast
        axes are supported (e.g. band, spinor) as long as the spatial
        axes are the last three.
    ``kvecs_frac``: ``(n_k, 3)`` fractional k-vectors.
    ``sign``: ``+1`` for the ψ post-IFFT case; ``-1`` for the ζ pre-FFT
        case (``z_q,μ(r) = exp(-2πi q·r) · ζ_q,μ(r)``).
    """
    nx, ny, nz = (int(s) for s in fft_grid)
    fx = jnp.arange(nx, dtype=jnp.float64) / nx
    fy = jnp.arange(ny, dtype=jnp.float64) / ny
    fz = jnp.arange(nz, dtype=jnp.float64) / nz

    scale = jnp.complex128(int(sign) * 2j * jnp.pi)
    # 1D per-axis per-k factors.
    px = jnp.exp(scale * kvecs_frac[:, 0:1] * fx[None, :])    # (n_k, nx)
    py = jnp.exp(scale * kvecs_frac[:, 1:2] * fy[None, :])
    pz = jnp.exp(scale * kvecs_frac[:, 2:3] * fz[None, :])

    # Broadcast helpers: pad the k-axis with however many intermediate
    # axes ``box`` has between k and the spatial axes.
    n_mid = box.ndim - 4          # k + (mid axes) + (x, y, z) = ndim
    mid_shape = (1,) * n_mid
    px = px.reshape(px.shape[0], *mid_shape, nx, 1, 1)
    py = py.reshape(py.shape[0], *mid_shape, 1, ny, 1)
    pz = pz.reshape(pz.shape[0], *mid_shape, 1, 1, nz)

    return box * px * py * pz
