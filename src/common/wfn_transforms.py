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
from jax.experimental.shard_map import shard_map
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P


__all__ = [
    "to_box", "to_rbox", "to_rmu", "to_rchunk",
    "to_rmu_inner", "to_rchunk_inner",
    "apply_bloch_phase", "apply_bloch_phase_on_slice",
    "gflat_to_rmu",
    "accumulate_rchunk_to_gflat",
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
# Sharding helpers — every public transform requires a ``mesh`` argument.
# A 1×1 trivial mesh is the right thing to pass for single-device runs;
# there is no ``mesh is None`` branch anywhere downstream.
# ---------------------------------------------------------------------------

def _spec_of(psi: jax.Array) -> tuple:
    """Partition spec for ``psi``, always of length ``psi.ndim``.

    Returns ``psi.sharding.spec`` padded with trailing ``None`` if
    JAX's ``PartitionSpec`` trimmed them off, else an all-None tuple
    (fully replicated) for inputs without a ``NamedSharding`` —
    e.g. single-device test arrays whose default sharding is
    ``SingleDeviceSharding``.
    """
    sh = getattr(psi, "sharding", None)
    if isinstance(sh, NamedSharding):
        spec = tuple(sh.spec)
        if len(spec) < psi.ndim:
            spec = spec + (None,) * (psi.ndim - len(spec))
        return spec
    return (None,) * psi.ndim


def _output_sharding(psi: jax.Array, mesh: Mesh, n_extra_axes: int) -> NamedSharding:
    """``NamedSharding`` for an output that mirrors psi's ``(n_k, nb, ns)``
    prefix and inserts ``n_extra_axes`` replicated axes after it.  The
    band axis is preserved; the leading and spinor axes are replicated."""
    spec = _spec_of(psi)
    band_spec = spec[1] if len(spec) >= 2 else None
    return NamedSharding(
        mesh, P(None, band_spec, None, *([None] * n_extra_axes)))


def _maybe_constrain(arr: jax.Array, sharding: NamedSharding) -> jax.Array:
    return jax.lax.with_sharding_constraint(arr, sharding)


# ---------------------------------------------------------------------------
# Local-FFT helper — the one FFT primitive used in this module.
# ---------------------------------------------------------------------------
#
# Plain ``jnp.fft.(i)fftn`` on a sharded tensor lets XLA's planner do
# whatever it likes — at CrI3 6×6×1 80 Ry it inserts an all-gather and
# emits a global FFT, blowing past the per-rank HBM ceiling (the
# original 121 GB OOM in ``to_rmu``).  ``make_sharded_*fftn_3d`` wraps
# cuFFT in a shard_map so the FFT runs per-rank-locally with no
# resharding; every FFT in this module goes through ``_local_box_fft``.

def _local_box_fft(psi: jax.Array, mesh: Mesh, *, kind: str, norm: str):
    """Sharded local FFT for the box ``psi.shape[:-1] + (nx, ny, nz)``.

    Returns a callable ``f(box) -> (i)fftn(box, axes=(-3, -2, -1))`` whose
    output sharding preserves psi's leading layout with the three FFT
    axes replicated.  ``kind`` is ``'fftn'`` or ``'ifftn'``.  The caller
    is responsible for the mesh — pass a 1×1 trivial mesh for
    single-device runs.
    """
    from common.fft_helpers import (
        make_sharded_ifftn_3d, make_sharded_fftn_3d)
    spec = P(*_spec_of(psi)[:-1], None, None, None)
    factory = (make_sharded_ifftn_3d if kind == 'ifftn'
               else make_sharded_fftn_3d)
    return factory(mesh, spec, spec, norm=norm, axes=(-3, -2, -1))


# ---------------------------------------------------------------------------
# Sharding signature key — used to keep the jit caches small and stable.
# ---------------------------------------------------------------------------

def _sharding_key(psi: jax.Array) -> tuple:
    """Hashable signature of psi's sharding (mesh identity + spec).

    Used as part of the jit-cache key for the public transforms so two
    arrays with identical mesh + PartitionSpec hit the same compiled
    XLA module, while a different mesh forces a fresh compile.

    ``spec`` is normalized to ``psi.ndim`` length: JAX sometimes trims
    trailing ``None`` entries on PartitionSpec (so ``P(None, ('x','y'))``
    and ``P(None, ('x','y'), None)`` are semantically equal but compare
    as different tuples).  Without the normalization the same kernel
    recompiles each time JAX hands back a trimmed spec — 11 extra
    ``_kernel`` compiles observed at MoS2 3×3 bispinor, adding ~7 s of
    wall time before this fix.  ``_spec_of`` above already does the
    same normalization for the in-body sharding derivation."""
    return (id(getattr(getattr(psi, "sharding", None), "mesh", None)),
            _spec_of(psi))


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
    *,
    mesh: Mesh,
) -> jax.Array:
    """Scatter G-flat ψ into the FFT box.

    Sharding (band axis on ``('x','y')`` or replicated) is preserved.
    ``g_index`` (output of :meth:`WfnLoader.box_index`) uses sentinel
    ``ngkmax`` to flag empty FFT-box cells (zero on gather).
    """
    ngkmax = int(psi.shape[-1])
    fft_grid_t = tuple(int(s) for s in fft_grid)
    out_sharding = _output_sharding(psi, mesh, n_extra_axes=3)
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
    mesh: Mesh,
    norm: str = "backward",
    kvecs_frac: np.ndarray | jax.Array | None = None,
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
    out_sharding = _output_sharding(psi, mesh, n_extra_axes=3)
    key = (psi.shape, tuple(g_index.shape), ngkmax, fft_grid_t, norm,
           kvecs_shape, _sharding_key(psi), out_sharding)
    fn = _RBOX_KERNEL_CACHE.get(key)
    if fn is None:
        ifftn = _local_box_fft(psi, mesh, kind='ifftn', norm=norm)
        if kvecs_frac is None:
            @jax.jit
            def fn(psi_, g_index_):
                box = _box_kernel(psi_, g_index_, ngkmax=ngkmax)
                return _maybe_constrain(ifftn(box), out_sharding)
            _RBOX_KERNEL_CACHE[key] = fn
        else:
            @jax.jit
            def fn(psi_, g_index_, kvecs_):
                box = _box_kernel(psi_, g_index_, ngkmax=ngkmax)
                rb = apply_bloch_phase(ifftn(box), kvecs_, fft_grid_t)
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
    mesh: Mesh,
    norm: str = "backward",
    kvecs_frac: np.ndarray | jax.Array | None = None,
) -> jax.Array:
    """ψ in r-space at the centroid FFT-grid indices ``r_mu``.

    ``r_mu`` is ``(n_rmu, 3)`` int32 (positions in ``[0, fft_grid[a])``);
    other args as :func:`to_rbox`.  Output ``(n_k, nb, nspinor, n_rmu)``
    with band-axis sharding preserved.
    """
    ngkmax = int(psi.shape[-1])
    fft_grid_t = tuple(int(s) for s in fft_grid)
    n_rmu = int(np.shape(r_mu)[0])
    kvecs_shape = (None if kvecs_frac is None
                   else tuple(int(s) for s in np.shape(kvecs_frac)))
    out_sharding = _output_sharding(psi, mesh, n_extra_axes=1)
    key = (psi.shape, tuple(g_index.shape), ngkmax, fft_grid_t, n_rmu,
           norm, kvecs_shape, _sharding_key(psi), out_sharding)
    fn = _RMU_KERNEL_CACHE.get(key)
    if fn is None:
        ifftn = _local_box_fft(psi, mesh, kind='ifftn', norm=norm)
        if kvecs_frac is None:
            @jax.jit
            def fn(psi_, g_index_, r_mu_):
                rb = ifftn(_box_kernel(psi_, g_index_, ngkmax=ngkmax))
                out = rb[:, :, :, r_mu_[:, 0], r_mu_[:, 1], r_mu_[:, 2]]
                return _maybe_constrain(out, out_sharding)
            _RMU_KERNEL_CACHE[key] = fn
        else:
            @jax.jit
            def fn(psi_, g_index_, r_mu_, kvecs_):
                rb = ifftn(_box_kernel(psi_, g_index_, ngkmax=ngkmax))
                rb = apply_bloch_phase(rb, kvecs_, fft_grid_t)
                out = rb[:, :, :, r_mu_[:, 0], r_mu_[:, 1], r_mu_[:, 2]]
                return _maybe_constrain(out, out_sharding)
            _RMU_KERNEL_CACHE[key] = fn
    g_index_j = jnp.asarray(g_index, dtype=jnp.int32)
    r_mu_j = jnp.asarray(r_mu, dtype=jnp.int32)
    if kvecs_frac is None:
        return fn(psi, g_index_j, r_mu_j)
    return fn(psi, g_index_j, r_mu_j,
              jnp.asarray(kvecs_frac, dtype=jnp.float64))


def to_rchunk_inner(
    psi: jax.Array,
    g_index: jax.Array,
    fft_grid: Sequence[int],
    r0,
    r_len: int,
    *,
    norm: str = "backward",
    kvecs_frac: jax.Array | None = None,
) -> jax.Array:
    """Per-rank-local body of :func:`to_rchunk`: G-flat → FFT-box → IFFT
    → r-slice → optional Bloch phase.

    No ``shard_map`` wrapper.  Callable from inside another shard_map's
    body or a ``lax.scan`` body — the caller is responsible for any
    sharding context.  Inputs and outputs are all per-rank-local arrays.

    Path D scaffolding (see
    ``reports/zeta_rchunk_memory_model_2026-05-13/agent_2_structural_fix.md``
    §4b).  Not yet wired into the production fit kernel — the consumer
    refactor (§4c, rewrite of ``c_q_from_psi_sm`` / ``z_q_from_psi_sm``
    with a ``lax.scan`` over bcs inside their shard_map bodies) is
    deferred to a follow-up session.  This helper is checked in early
    so it can be unit-tested independently.

    Mathematically identical to the body of :func:`to_rchunk`; the only
    difference is that this version does not enter a shard_map.

    Inputs
    ------
    psi
        Shape ``(..., ngkmax)`` c128 — caller's responsibility to have
        already partitioned data across ranks if running under a
        shard_map.  The leading axes must contain exactly 3 dims before
        the trailing ``ngkmax`` so the post-IFFT reshape lands at
        ``(..., n_rtot)`` — typically ``(nk_local, nb_local, ns,
        ngkmax)``.
    g_index
        Shape ``(..., ngkmax)`` int32 — flat box indices for each
        G-vector per k.  Same broadcast contract as :func:`to_rchunk`.
    fft_grid
        ``(nx, ny, nz)``.
    r0
        Python int or traced int32 scalar — flat-r start index.
    r_len
        Static int — width of the r slab.
    norm
        FFT normalization, same conventions as ``jnp.fft.ifftn``.
    kvecs_frac
        Optional ``(..., 3)`` float64 — when provided, the Bloch
        phase ``exp(+2πi k·r)`` is applied on the sliced slab via
        :func:`apply_bloch_phase_on_slice`.

    Output
    ------
    jax.Array
        Shape ``(..., r_len)`` c128.
    """
    ngkmax = int(psi.shape[-1])
    fft_grid_t = tuple(int(s) for s in fft_grid)
    nx, ny, nz = fft_grid_t
    n_rtot = nx * ny * nz
    r_len_i = int(r_len)

    box = _box_kernel(psi, g_index, ngkmax=ngkmax)
    rb = jnp.fft.ifftn(box, axes=(-3, -2, -1), norm=norm)
    # Reshape (..., nx, ny, nz) → (..., n_rtot).  Same contract as
    # to_rchunk._local_rchunk: assumes 3 leading axes before the spatial.
    rb_flat = rb.reshape(*rb.shape[:3], n_rtot)
    slab = jax.lax.dynamic_slice_in_dim(rb_flat, r0, r_len_i, axis=-1)
    if kvecs_frac is not None:
        slab = apply_bloch_phase_on_slice(
            slab, kvecs_frac, fft_grid_t, r0, r_len_i)
    return slab


def to_rchunk(
    psi: jax.Array,
    g_index: np.ndarray | jax.Array,
    fft_grid: Sequence[int],
    r0,
    r_len: int,
    *,
    mesh: Mesh,
    norm: str = "backward",
    kvecs_frac: np.ndarray | jax.Array | None = None,
) -> jax.Array:
    """ψ in r-space on a contiguous flat-r slab ``[r0, r0 + r_len)``.

    Flat-r convention: ``r_flat = rx * ny * nz + ry * nz + rz``.  ``r0``
    may be a Python int (bounds-checked) or a traced scalar (caller's
    responsibility).

    The full G-flat gather → FFT-box → IFFT → r-slice (→ Bloch phase)
    pipeline runs inside one ``shard_map`` region.  Keeping it
    inside the manual per-rank region prevents XLA SPMD from
    reconstructing the full logical band axis between ``PsiGStore``'s
    ``io_callback`` and the local FFT — verified to remove ~506 MiB
    all-gathers from the HLO at MoS2 3×3 / 4×A100.
    """
    ngkmax = int(psi.shape[-1])
    fft_grid_t = tuple(int(s) for s in fft_grid)
    nx, ny, nz = fft_grid_t
    n_rtot = nx * ny * nz
    r_len_i = int(r_len)
    if isinstance(r0, (int, np.integer)):
        if r0 < 0 or int(r0) + r_len_i > n_rtot:
            raise ValueError(
                f"to_rchunk: [{int(r0)}, {int(r0) + r_len_i}) "
                f"out of [0, {n_rtot})")

    psi_spec = P(*_spec_of(psi))
    out_spec = tuple(_output_sharding(psi, mesh, n_extra_axes=1).spec)
    kvecs_shape = (None if kvecs_frac is None
                   else tuple(int(s) for s in np.shape(kvecs_frac)))
    key = (psi.shape, tuple(g_index.shape), ngkmax, fft_grid_t, r_len_i,
           norm, kvecs_shape, _sharding_key(psi), out_spec, id(mesh))
    fn = _RCHUNK_KERNEL_CACHE.get(key)
    if fn is None:
        if kvecs_frac is None:
            @partial(
                shard_map,
                mesh=mesh,
                in_specs=(psi_spec, P(None, None, None, None), P()),
                out_specs=P(*out_spec),
                check_rep=False,
            )
            def _local_rchunk(psi_l, g_index_l, r0_l):
                box_l = _box_kernel(psi_l, g_index_l, ngkmax=ngkmax)
                rb_l = jnp.fft.ifftn(box_l, axes=(-3, -2, -1), norm=norm)
                rb_flat_l = rb_l.reshape(*rb_l.shape[:3], n_rtot)
                return jax.lax.dynamic_slice_in_dim(
                    rb_flat_l, r0_l, r_len_i, axis=-1)

            @jax.jit
            def fn(psi_, g_index_, r0_):
                return _local_rchunk(psi_, g_index_, r0_)
        else:
            @partial(
                shard_map,
                mesh=mesh,
                in_specs=(psi_spec, P(None, None, None, None), P(),
                          P(None, None)),
                out_specs=P(*out_spec),
                check_rep=False,
            )
            def _local_rchunk(psi_l, g_index_l, r0_l, kvecs_l):
                # Phase-after-slice: IFFT → flatten → slice → phase on
                # the r_len-cell slab (not on the full nx·ny·nz box).
                box_l = _box_kernel(psi_l, g_index_l, ngkmax=ngkmax)
                rb_l = jnp.fft.ifftn(box_l, axes=(-3, -2, -1), norm=norm)
                rb_flat_l = rb_l.reshape(*rb_l.shape[:3], n_rtot)
                slab_l = jax.lax.dynamic_slice_in_dim(
                    rb_flat_l, r0_l, r_len_i, axis=-1)
                return apply_bloch_phase_on_slice(
                    slab_l, kvecs_l, fft_grid_t, r0_l, r_len_i)

            @jax.jit
            def fn(psi_, g_index_, r0_, kvecs_):
                return _local_rchunk(psi_, g_index_, r0_, kvecs_)
        _RCHUNK_KERNEL_CACHE[key] = fn

    g_index_j = jnp.asarray(g_index, dtype=jnp.int32)
    r0_arg = (jnp.int32(int(r0)) if isinstance(r0, (int, np.integer)) else r0)
    if kvecs_frac is None:
        return fn(psi, g_index_j, r0_arg)
    return fn(psi, g_index_j, r0_arg,
              jnp.asarray(kvecs_frac, dtype=jnp.float64))



# ---------------------------------------------------------------------------
# G-flat → r-centroid helpers.  ``to_rmu_inner`` is the pure-jax body of
# :func:`to_rmu` (callable from inside another shard_map or scan body),
# and ``gflat_to_rmu`` is the bc-scan-inside-shard_map twin of
# :func:`gflat_to_rchunk` for the centroid-sample direction.
# ---------------------------------------------------------------------------
#
# Same Defect 1 / Defect 3 family as the r-slab pair: the legacy
# centroid-load path bc-loops outside ``to_rmu`` and ``to_rmu`` itself
# materialises an unsharded ``c128[nk, band_chunk, ns, nx, ny, nz]``
# FFT box on every rank (Peak A in the planner — single slot, but the
# §0 principle treats unsharded-on-every-rank as a violation regardless
# of slot count).  ``gflat_to_rmu`` collapses both the bc-loop AND the
# inner unsharded FFT box into one shard_map + lax.scan, mirroring
# :func:`gflat_to_rchunk` modulo the r-slab → centroid-sample swap.

def to_rmu_inner(
    psi: jax.Array,
    g_index: jax.Array,
    fft_grid: Sequence[int],
    r_mu: jax.Array,
    *,
    norm: str = "backward",
    kvecs_frac: jax.Array | None = None,
) -> jax.Array:
    """Per-rank-local body of :func:`to_rmu`: G-flat → FFT-box → IFFT
    → centroid sample → optional Bloch phase.

    No ``shard_map`` wrapper.  Callable from inside another shard_map's
    body or a ``lax.scan`` body — the caller is responsible for any
    sharding context.  Inputs and outputs are all per-rank-local arrays.

    Mirror of :func:`to_rchunk_inner` for the centroid-sample direction.
    Mathematically identical to the body of :func:`to_rmu`; the only
    difference is that this version does not enter a shard_map.

    Inputs
    ------
    psi
        Shape ``(..., ngkmax)`` c128 — caller's responsibility to have
        already partitioned data across ranks if running under a
        shard_map.  Leading axes must contain exactly 3 dims before the
        trailing ``ngkmax`` so the post-IFFT reshape lands at
        ``(..., nx, ny, nz)`` — typically ``(nk_local, nb_local, ns,
        ngkmax)``.
    g_index
        Shape ``(..., ngkmax)`` int32 — flat box indices for each
        G-vector per k.  Same broadcast contract as :func:`to_rmu`.
    fft_grid
        ``(nx, ny, nz)``.
    r_mu
        Shape ``(n_rmu, 3)`` int32 — FFT-grid coordinates of the
        centroid sample points.
    norm
        FFT normalization, same conventions as ``jnp.fft.ifftn``.
    kvecs_frac
        Optional ``(n_k, 3)`` float64 — when provided, the Bloch
        phase ``exp(+2πi k·r)`` is applied to the full FFT box via
        :func:`apply_bloch_phase` before the centroid gather (same as
        :func:`to_rmu`'s body).

    Output
    ------
    jax.Array
        Shape ``(..., n_rmu)`` c128.
    """
    ngkmax = int(psi.shape[-1])
    fft_grid_t = tuple(int(s) for s in fft_grid)
    box = _box_kernel(psi, g_index, ngkmax=ngkmax)
    rb = jnp.fft.ifftn(box, axes=(-3, -2, -1), norm=norm)
    if kvecs_frac is not None:
        rb = apply_bloch_phase(rb, kvecs_frac, fft_grid_t)
    # Gather centroid cells: trailing (nx, ny, nz) → (n_rmu,).
    out = rb[..., r_mu[:, 0], r_mu[:, 1], r_mu[:, 2]]
    return out


# ---------------------------------------------------------------------------
# Structural fix: shard_map + scan over chunks of the flat (nk · nb_local)
# axis, mirroring :func:`gflat_to_rchunk` but with the centroid-sample
# gather in place of the r-slab slice.  Inside the scan body XLA's
# allocator aliases the per-iter FFT box across iters, so the slot count
# for that buffer-class collapses to 1 — and inside the shard_map the
# FFT box is per-rank-local (NOT replicated on every rank) so the
# unsharded-FFT-box violation in the legacy ``to_rmu`` path is closed.

_GFLAT_TO_RMU_CACHE: dict = {}


def gflat_to_rmu(
    psi_G: jax.Array,
    g_index: np.ndarray | jax.Array,
    r_mu: np.ndarray | jax.Array,
    *,
    mesh: Mesh,
    fft_grid: Sequence[int],
    kvecs_frac: np.ndarray | jax.Array | None = None,
    norm: str = "backward",
    chunk_size: int | None = None,
) -> jax.Array:
    """ψ(G-flat) → ψ at centroid grid points, fused over all (k, n).

    Shapes / shardings (mesh = ``('x', 'y')`` of size ``P = p_x · p_y``)::

        psi_G   : (nk, nb_total, ns, ngkmax)  P(None, ('x','y'), None, None)
        return  : (nk, nb_total, ns, n_rmu)   P(None, ('x','y'), None, None)

    ``nb_total`` must be divisible by ``mesh.size``.  Each rank owns a
    ``nb_local = nb_total / P`` block of bands across the full
    ``(nk, ns, ngkmax)`` extent and writes the same band block to the
    output.

    Algorithm — inside a single ``shard_map`` over ``('x','y')``:

      1. Flatten per-rank ``(nk, nb_local) → N = nk · nb_local`` rows so
         every row is a single ``(k, n)`` pair.
      2. Zero-pad ``N → ⌈N / cs⌉ · cs`` so the chunk count is exact.
      3. ``lax.scan`` over chunks of ``cs`` rows.  Each iteration:
         a. ``k_row[cs] = (i·cs + arange(cs)) // nb_local`` —
            which k each row belongs to.  Clipped to ``[0, nk)``;
            padding rows land on k = nk - 1 but their data is zero
            (zero-pad) so the centroid samples they produce are
            zero and get truncated in ``out_flat[:N]``.
         b. ``_box_kernel(sub[cs, 1, ns, ngkmax], g_index[k_row])``
            scatters G-sphere coeffs into a per-row FFT box
            ``(cs, 1, ns, nx, ny, nz)``.  Singleton ``nb`` axis is
            squeezed.
         c. ``jnp.fft.ifftn`` on the trailing 3 axes — per-rank-local
            cuFFT, no resharding.
         d. Gather centroid cells: ``rb[:, :, r_mu[:,0], r_mu[:,1],
            r_mu[:,2]]`` → ``(cs, ns, n_rmu)``.
         e. Optional per-row Bloch phase ``exp(+2πi k·r_mu)`` applied
            on the *gathered cells only* (not on the full box) —
            ``apply_bloch_phase``-equivalent under the gather, but
            scratch drops from ``(cs · n_rtot)`` to ``(cs · n_rmu)``.
         f. ``dynamic_update_slice_in_dim`` writes the row block into
            ``out_flat`` at offset ``i·cs``.

    Chunking is on the flat ``(k · n_local)`` axis so the chunk size
    is a free integer — no divisibility constraint on either nk or
    nb_local.  Defaults to one-shot (``cs = N``); the scan compiles to a
    single iteration that XLA folds away.  Per-iteration FFT-box
    transient is ``cs · ns · n_rtot · 16`` bytes — choose ``chunk_size``
    to bound this against the per-rank HBM budget.

    Parameters
    ----------
    psi_G
        ``(nk, nb_total, ns, ngkmax)`` c128, band-flat-sharded.
    g_index
        ``(nk, nx, ny, nz)`` int32 — flat-FFT-box indices.  Sentinel
        value ``ngkmax`` flags empty box cells (zero on gather).
        Replicated.
    r_mu
        ``(n_rmu, 3)`` int32 — FFT-grid coordinates of the centroid
        sample points.  Replicated.
    mesh
        Process mesh with named axes ``'x'`` and ``'y'``.
    fft_grid
        Static ``(nx, ny, nz)``.
    kvecs_frac
        Optional ``(nk, 3)`` fractional k-vectors.  When given, the
        gathered samples are multiplied by ``exp(+2πi k·r_mu)`` per
        ``(k, r_mu)`` pair (separable in x/y/z; pre-computed once per
        k).  ``None`` skips the phase.
    norm
        Forwarded to :func:`jnp.fft.ifftn`.  Defaults to ``"backward"``
        to match the legacy :func:`to_rmu` default; centroid-load
        callers typically pass ``"ortho"``.
    chunk_size
        Rows per scan iteration along the flat ``(k · n_local)``
        axis.  Default ``None`` ⇒ one-shot.  Memory bound:
        ``chunk_size · ns · n_rtot · 16 B`` for the per-iteration FFT
        box.

    Returns
    -------
    ``(nk, nb_total, ns, n_rmu)`` c128 with ``P(None, ('x','y'),
    None, None)`` sharding.
    """
    fft_grid_t = tuple(int(s) for s in fft_grid)
    nx, ny, nz = fft_grid_t
    n_rtot = nx * ny * nz
    nk        = int(psi_G.shape[0])
    nb_total  = int(psi_G.shape[1])
    ns        = int(psi_G.shape[2])
    ngkmax    = int(psi_G.shape[3])
    r_mu_arr  = np.ascontiguousarray(np.asarray(r_mu, dtype=np.int32))
    if r_mu_arr.ndim != 2 or int(r_mu_arr.shape[1]) != 3:
        raise ValueError(
            f"gflat_to_rmu: r_mu must be (n_rmu, 3); got shape "
            f"{r_mu_arr.shape}.")
    n_rmu     = int(r_mu_arr.shape[0])
    p_prod    = int(np.prod([mesh.shape[a] for a in mesh.axis_names]))
    if nb_total % p_prod != 0:
        raise ValueError(
            f"gflat_to_rmu: nb_total={nb_total} not divisible by "
            f"mesh.size={p_prod}.")
    nb_local = nb_total // p_prod
    N        = nk * nb_local

    g_arr = np.asarray(g_index, dtype=np.int32)
    if g_arr.ndim != 4 or int(g_arr.shape[0]) != nk:
        raise ValueError(
            f"gflat_to_rmu: g_index must be (nk, nx, ny, nz); got "
            f"shape {g_arr.shape}, expected nk={nk}.")
    if tuple(int(s) for s in g_arr.shape[1:]) != fft_grid_t:
        raise ValueError(
            f"gflat_to_rmu: g_index trailing shape {g_arr.shape[1:]} "
            f"≠ fft_grid {fft_grid_t}.")

    # r_mu range check (Python int values; replicated, OK to validate
    # at trace time).
    if (np.any(r_mu_arr[:, 0] < 0) or np.any(r_mu_arr[:, 0] >= nx)
            or np.any(r_mu_arr[:, 1] < 0) or np.any(r_mu_arr[:, 1] >= ny)
            or np.any(r_mu_arr[:, 2] < 0) or np.any(r_mu_arr[:, 2] >= nz)):
        raise ValueError(
            f"gflat_to_rmu: r_mu has out-of-range coords for "
            f"fft_grid {fft_grid_t}.")

    cs       = int(chunk_size if chunk_size else N)
    n_chunks = (N + cs - 1) // cs
    pad_N    = n_chunks * cs - N

    if kvecs_frac is None:
        kvecs_shape = None
        kvecs_id = None
    else:
        kvecs_arr = np.ascontiguousarray(np.asarray(kvecs_frac, dtype=np.float64))
        kvecs_shape = tuple(int(s) for s in kvecs_arr.shape)
        # Content-hash kvecs — same lesson as gflat_to_rchunk: the
        # per-k phase tables (phx/phy/phz) bake into the cached
        # closure, so shape-only keying would silently reuse stale
        # tables when a different caller passes new kvecs_frac.
        kvecs_id = hash(kvecs_arr.tobytes())
    g_index_id = hash(g_arr.tobytes())
    r_mu_id    = hash(r_mu_arr.tobytes())

    key = (
        tuple(int(s) for s in psi_G.shape),
        fft_grid_t, n_rmu, r_mu_id, ngkmax, g_index_id,
        norm, kvecs_shape, kvecs_id, cs, n_chunks, pad_N,
        _sharding_key(psi_G),
    )
    fn = _GFLAT_TO_RMU_CACHE.get(key)
    if fn is None:
        # Per-k phase tables and centroid-coord constants baked into the
        # closure.  r_mu is replicated so we can index the per-row
        # phases by r_mu directly inside the scan body.
        g_index_c = jnp.asarray(g_arr, dtype=jnp.int32)
        r_mu_c    = jnp.asarray(r_mu_arr, dtype=jnp.int32)
        if kvecs_frac is not None:
            kv = jnp.asarray(np.asarray(kvecs_frac), dtype=jnp.float64)
            # Forward Bloch phase: sign = +1 (post-IFFT, ψ = exp(+2πi k·r)·u).
            _ph = lambda k_axis, n: jnp.exp(
                +2j * jnp.pi * k_axis[:, None] * (jnp.arange(n) / n)[None, :])
            phx, phy, phz = _ph(kv[:, 0], nx), _ph(kv[:, 1], ny), _ph(kv[:, 2], nz)
            # Per-centroid 1D-phase columns: (nk, n_rmu) each.
            phx_rmu_all = phx[:, r_mu_c[:, 0]]    # (nk, n_rmu)
            phy_rmu_all = phy[:, r_mu_c[:, 1]]
            phz_rmu_all = phz[:, r_mu_c[:, 2]]
        else:
            phx_rmu_all = phy_rmu_all = phz_rmu_all = None

        in_spec  = P(None, ('x', 'y'), None, None)
        out_spec = P(None, ('x', 'y'), None, None)

        @partial(shard_map, mesh=mesh,
                 in_specs=(in_spec,),
                 out_specs=out_spec,
                 check_rep=False)
        def _kernel(psi_):
            # Per-rank: (nk, nb_local, ns, ngkmax).
            psi_flat = psi_.reshape(N, ns, ngkmax)
            if pad_N:
                psi_flat = jnp.pad(
                    psi_flat, ((0, pad_N), (0, 0), (0, 0)))
            out_flat = jnp.zeros(
                (N + pad_N, ns, n_rmu), dtype=psi_.dtype)

            def body(out, i):
                i0    = i * cs
                sub   = jax.lax.dynamic_slice_in_dim(
                    psi_flat, i0, cs, axis=0)            # (cs, ns, ngkmax)
                k_row = jnp.clip(
                    (i0 + jnp.arange(cs)) // nb_local, 0, nk - 1)  # (cs,)
                # Singleton-nb reshape so _box_kernel's (n_k, nb, ns,
                # ngkmax) contract takes (cs, 1, ns, ngkmax) per row.
                # Per-row g_index gather: (cs, nx, ny, nz).
                sub4 = sub.reshape(cs, 1, ns, ngkmax)
                g_per_row = g_index_c[k_row]
                box = _box_kernel(
                    sub4, g_per_row, ngkmax=ngkmax)      # (cs, 1, ns, nx, ny, nz)
                box = box.reshape(cs, ns, nx, ny, nz)
                rb = jnp.fft.ifftn(box, axes=(-3, -2, -1), norm=norm)
                # Centroid gather — (cs, ns, n_rmu).
                samples = rb[:, :, r_mu_c[:, 0], r_mu_c[:, 1], r_mu_c[:, 2]]
                if phx_rmu_all is not None:
                    # Per-row Bloch phase at the gathered centroid
                    # cells — apply_bloch_phase is multiplicative on the
                    # spatial axes, so applying it post-gather is
                    # algebraically identical to applying it on the
                    # full FFT box before gather (cf. gflat_to_rchunk's
                    # phase-on-slice pattern).
                    phx_q = phx_rmu_all[k_row]           # (cs, n_rmu)
                    phy_q = phy_rmu_all[k_row]
                    phz_q = phz_rmu_all[k_row]
                    samples = samples * (phx_q * phy_q * phz_q)[:, None, :]
                return jax.lax.dynamic_update_slice_in_dim(
                    out, samples, i0, axis=0), None

            out_flat, _ = jax.lax.scan(
                body, out_flat, jnp.arange(n_chunks, dtype=jnp.int32))
            if pad_N:
                out_flat = out_flat[:N]
            return out_flat.reshape(nk, nb_local, ns, n_rmu)

        fn = jax.jit(_kernel)
        _GFLAT_TO_RMU_CACHE[key] = fn

    return fn(psi_G)


# ---------------------------------------------------------------------------
# rchunk → G-flat partial-sum accumulator
# ---------------------------------------------------------------------------
#
# Mirror of :func:`to_rchunk` in the opposite direction.  Used by the
# new G-flat zeta writer (Phase C of PLAN_zeta_g_flat_migration.md):
# for each r-chunk produced by the ζ solve, the caller feeds the
# slab back through this function which adds its FFT-G-sphere
# contribution into a persistent ``gflat_acc`` buffer.  After the loop
# over r-chunks completes, ``gflat_acc`` holds the full G-flat ζ_q.
#
# Two design choices worth preserving:
#
# 1. **Phase-on-slice** (mirrors :func:`apply_bloch_phase_on_slice` for
#    the inverse direction with ``sign=-1``).  The full FFT box is
#    NEVER materialised on the rchunk side — only ``r_len`` cells get
#    a phase multiply.  The box DOES temporarily exist around the
#    forward FFT itself (cuFFT needs a dense box); that's the only
#    big transient and it's scoped to one chunked call.
#
# 2. **Donated accumulator**.  ``gflat_acc`` is the only persistent
#    buffer; donation makes the ``acc + contribution`` an in-place
#    add under jit.
#
# Math: for each (q, μ_local) row of ``rchunk`` of size ``r_len``,
#
#     ζ_G[q, μ, G_sph] = Σ_r  exp(-2πi q·r) ζ_r[q, μ, r]  e^{-2πi G·r}
#                     = FFT_{r→G}( exp(-2πi q·r) ζ_r[q, μ, r] )[G_sph]
#
# Linearity over r means each r-chunk is an additive contribution:
#     ζ_G += FFT_{r→G}( phase · pad_to_full(rchunk_slab) )[G_sph]
# which is what this function accumulates.

_RCHUNK_TO_GFLAT_CACHE: dict = {}


def accumulate_rchunk_to_gflat(
    rchunk: jax.Array,
    gflat_acc: jax.Array,
    *,
    mesh: Mesh,
    fft_grid: Sequence[int],
    r0,
    sphere_idx: np.ndarray | jax.Array,
    qvec_frac: np.ndarray | jax.Array | None = None,
    norm: str = "backward",
    chunk_size: int | None = None,
) -> jax.Array:
    """Add ``FFT(pad(phase(rchunk)))[sphere_idx]`` into ``gflat_acc``.

    Shapes / shardings (mesh = ``('x', 'y')`` of size ``P = p_x · p_y``)::

        rchunk    : (n_q, n_rmu_padded, r_len)   P(None, ('x','y'), None)
        gflat_acc : (n_q, n_rmu_padded, ngkmax)  P(None, ('x','y'), None)

    ``n_rmu_padded`` must be a multiple of ``mesh.size`` (rounded up at
    :class:`common.meta.Meta` construction).  Each rank owns a
    ``n_mu_local = n_rmu_padded / P`` block of μ-rows over the full
    ``(n_q, r_len)`` extent.

    Algorithm — inside a single ``shard_map`` over ``('x','y')``:

      1. Flatten ``(n_q, n_mu_local) → N = n_q · n_mu_local`` rows.
      2. Zero-pad ``N → ⌈N / cs⌉ · cs`` so the chunk count is exact.
      3. ``lax.scan`` over chunks of ``cs`` rows.  Each iteration:
         a. ``q_row[cs] = (i·cs + arange(cs)) // n_mu_local`` —
            which q each row belongs to.  Clipped to ``[0, n_q)``;
            padding rows land on q = n_q - 1 but their data is zero
            (zero-pad) so the contrib they produce is zero — no
            contamination of ``acc``.
         b. Slab → FFT box via ``dynamic_update_slice_in_dim`` at
            offset ``r0``.
         c. Per-q Bloch phase, if ``qvec_frac`` given: separable
            ``exp(-2πi q · r)`` factors, pre-computed per q at trace
            time, gathered per row by ``q_row``.
         d. ``jnp.fft.fftn`` on the trailing 3 axes — data is
            per-rank-local on the entire FFT box (FFT axes are
            replicated by sharding contract), so plain ``fftn`` runs
            local cuFFT with no resharding.
         e. ``jnp.take_along_axis`` gathers the per-q sphere indices
            (``sphere_idx[q_row]``) along the trailing flat-r axis.
         f. Accumulate into ``acc`` at offset ``i·cs``.

    Chunking is on the flat ``(q · μ_local)`` axis so the chunk size
    is a free integer — no divisibility constraint on either n_q or
    n_mu_local.  Defaults to one-shot (``cs = N``); the scan compiles
    to a single iteration that XLA folds away.

    Parameters
    ----------
    rchunk
        ``(n_q, n_rmu_padded, r_len)`` c128, μ-flat-sharded.
    gflat_acc
        ``(n_q, n_rmu_padded, ngkmax)`` c128, μ-flat-sharded.
        Donated by the inner jit — its buffer is reused in place.
    mesh
        Process mesh with named axes ``'x'`` and ``'y'``.
    fft_grid
        Static ``(nx, ny, nz)``.
    r0
        Python int or jax-scalar — flat-r start of the slab in
        ``[0, nx·ny·nz)``.
    sphere_idx
        ``(n_q, ngkmax)`` int32 flat-FFT indices.  Every q has the
        same ``ngkmax`` axis length with potentially different index
        lists per q; pad slots within a row use a sentinel flat-FFT
        index whose coeffs the caller zeroes post-loop.
    qvec_frac
        Optional ``(n_q, 3)`` fractional q-vectors.  When given, the
        FFT box is multiplied by ``exp(-2πi q · r)`` per row
        (separable in x/y/z; pre-computed once per q).
    norm
        Forwarded to :func:`jnp.fft.fftn`.
    chunk_size
        Rows per scan iteration along the flat ``(q · μ_local)``
        axis.  Default ``None`` ⇒ one-shot.  Memory bound:
        ``chunk_size · n_rtot · 16 B`` for the per-iteration FFT box.

    Returns
    -------
    Updated ``gflat_acc`` (same shape, same sharding).
    """
    fft_grid_t = tuple(int(s) for s in fft_grid)
    nx, ny, nz = fft_grid_t
    n_rtot = nx * ny * nz
    n_q       = int(rchunk.shape[0])
    n_rmu_pad = int(rchunk.shape[1])
    r_len_i   = int(rchunk.shape[-1])
    p_prod    = int(np.prod([mesh.shape[a] for a in mesh.axis_names]))
    if n_rmu_pad % p_prod != 0:
        raise ValueError(
            f"accumulate_rchunk_to_gflat: n_rmu_padded={n_rmu_pad} not "
            f"divisible by mesh.size={p_prod}.")
    n_mu_local = n_rmu_pad // p_prod
    N          = n_q * n_mu_local

    sphere_arr = np.asarray(sphere_idx, dtype=np.int32)
    if sphere_arr.ndim != 2 or int(sphere_arr.shape[0]) != n_q:
        raise ValueError(
            f"accumulate_rchunk_to_gflat: sphere_idx must be (n_q, ngkmax); "
            f"got shape {sphere_arr.shape}, expected n_q={n_q}.")
    ngkmax  = int(sphere_arr.shape[-1])

    if isinstance(r0, (int, np.integer)) and (int(r0) < 0 or int(r0) + r_len_i > n_rtot):
        raise ValueError(
            f"accumulate_rchunk_to_gflat: r-slab "
            f"[{int(r0)}, {int(r0) + r_len_i}) out of [0, {n_rtot}).")

    cs       = int(chunk_size if chunk_size else N)
    n_chunks = (N + cs - 1) // cs
    pad_N    = n_chunks * cs - N

    qvec_shape = (None if qvec_frac is None
                  else tuple(int(s) for s in np.shape(qvec_frac)))
    # Content-hash qvec_frac so two callers with the same shape but
    # different q-grids don't silently collide on a cached fn whose
    # closure holds stale phx/phy/phz tables.  Mirrors the sphere_id
    # pattern below and gflat_to_rchunk's kvecs_id.  Masked in
    # production because q-grid is constant per run, but a latent
    # correctness hazard for any caller that varies qvec_frac
    # at fixed shape.
    qvec_id = (0 if qvec_frac is None
               else hash(np.asarray(qvec_frac, dtype=np.float64).tobytes()))
    sphere_id  = hash(sphere_arr.tobytes())

    key = (
        tuple(int(s) for s in rchunk.shape),
        tuple(int(s) for s in gflat_acc.shape),
        fft_grid_t, r_len_i, ngkmax, sphere_id,
        norm, qvec_shape, qvec_id, cs, n_chunks, pad_N,
        _sharding_key(rchunk), _sharding_key(gflat_acc),
    )
    fn = _RCHUNK_TO_GFLAT_CACHE.get(key)
    if fn is None:
        # Per-q tables baked into the closure as constants.
        sphere_c = jnp.asarray(sphere_arr, dtype=jnp.int32)
        if qvec_frac is not None:
            qv = jnp.asarray(np.asarray(qvec_frac), dtype=jnp.float64)
            _ph = lambda q_axis, n: jnp.exp(
                -2j * jnp.pi * q_axis[:, None] * (jnp.arange(n) / n)[None, :])
            phx, phy, phz = _ph(qv[:, 0], nx), _ph(qv[:, 1], ny), _ph(qv[:, 2], nz)
        else:
            phx = phy = phz = None

        # Sharding: μ is the only sharded axis on both rchunk and
        # gflat_acc.  P(None, ('x','y'), None) is enforced by the
        # caller; the shard_map below sees per-rank slabs.
        in_spec  = P(None, ('x', 'y'), None)
        out_spec = P(None, ('x', 'y'), None)

        @partial(shard_map, mesh=mesh,
                 in_specs=(in_spec, in_spec, P()),
                 out_specs=out_spec,
                 check_rep=False)
        def _kernel(rch_, acc_, r0_):
            # Per-rank: (n_q, n_mu_local, r_len) / (n_q, n_mu_local, ngkmax).
            rch_flat = rch_.reshape(N, r_len_i)
            acc_flat = acc_.reshape(N, ngkmax)
            if pad_N:
                rch_flat = jnp.pad(rch_flat, ((0, pad_N), (0, 0)))
                acc_flat = jnp.pad(acc_flat, ((0, pad_N), (0, 0)))

            # ----- Phase-on-slice setup (loop-invariant across the scan) -----
            # The per-q Bloch phase ``exp(-2πi q·r)`` is applied only to the
            # ``r_len`` slab cells (not to the zero-padded box) — saves
            # ``(n_rtot - r_len) / n_rtot`` of the phase-multiply traffic.
            # Decode r0_ + j into (rx, ry, rz) once outside ``body`` since
            # r0_ is loop-invariant within the scan (the scan iterates over
            # rows of the flat (q · μ_local) axis, not r-chunks).
            if phx is not None:
                r_idx_slab = r0_ + jnp.arange(r_len_i, dtype=jnp.int32)
                ny_nz = jnp.int32(ny * nz)
                rx_slab = r_idx_slab // ny_nz
                ry_slab = (r_idx_slab // jnp.int32(nz)) % jnp.int32(ny)
                rz_slab = r_idx_slab %  jnp.int32(nz)

            def body(acc, i):
                i0    = i * cs
                sub   = jax.lax.dynamic_slice_in_dim(rch_flat, i0, cs, axis=0)
                q_row = jnp.clip(
                    (i0 + jnp.arange(cs)) // n_mu_local, 0, n_q - 1)
                if phx is not None:
                    # Per-q Bloch phase on the slab only: gather (cs, r_len)
                    # from each axis's (n_q, n_*) table via the (cs,) q_row
                    # and the (r_len,) slab-cell indices.  XLA fuses these
                    # three gather+multiply ops into one pointwise pass.
                    phx_q = phx[q_row][:, rx_slab]     # (cs, r_len)
                    phy_q = phy[q_row][:, ry_slab]
                    phz_q = phz[q_row][:, rz_slab]
                    sub = sub * phx_q * phy_q * phz_q
                buf = jnp.zeros((cs, n_rtot), dtype=sub.dtype)
                buf = jax.lax.dynamic_update_slice_in_dim(buf, sub, r0_, axis=-1)
                box = buf.reshape(cs, nx, ny, nz)
                G = jnp.fft.fftn(box, axes=(-3, -2, -1), norm=norm).reshape(cs, n_rtot)
                contrib = jnp.take_along_axis(
                    G, sphere_c[q_row], axis=-1, mode='promise_in_bounds')
                acc_sub = jax.lax.dynamic_slice_in_dim(acc, i0, cs, axis=0)
                return jax.lax.dynamic_update_slice_in_dim(
                    acc, acc_sub + contrib, i0, axis=0), None

            acc_flat, _ = jax.lax.scan(
                body, acc_flat, jnp.arange(n_chunks, dtype=jnp.int32))
            if pad_N:
                acc_flat = acc_flat[:N]
            return acc_flat.reshape(n_q, n_mu_local, ngkmax)

        fn = jax.jit(_kernel, donate_argnums=(1,))
        _RCHUNK_TO_GFLAT_CACHE[key] = fn

    r0_arg = (jnp.int32(int(r0)) if isinstance(r0, (int, np.integer)) else r0)
    return fn(rchunk, gflat_acc, r0_arg)


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


def apply_bloch_phase_on_slice(
    slab: jax.Array,
    kvecs_frac: jax.Array,
    fft_grid: tuple[int, int, int],
    r0,
    r_len: int,
    *,
    sign: int = 1,
) -> jax.Array:
    """``slab × exp(sign·2πi k·r)`` over a contiguous flat-r slab.

    Flat-r convention matches :func:`to_rchunk`:
    ``r_flat = rx · ny · nz + ry · nz + rz``.

    Where :func:`apply_bloch_phase` builds the phase over the full FFT
    box and relies on the caller to slice the result, this helper
    builds the phase only on the requested slab ``[r0, r0 + r_len)``.
    Mathematically identical (IFFT + multiply commutes with slicing
    along r); operationally important when the slab is much smaller
    than the full box — pulls per-r-cell work from
    ``n_k × nx · ny · nz`` down to ``n_k × r_len``.

    ``slab``: trailing shape ``(..., r_len)``.  Sharding preserved.
    ``r0``: Python int or a jax scalar (traced).  When traced, callers
        are responsible for the bounds check.
    ``r_len``: static int — slab length.
    """
    nx, ny, nz = (int(s) for s in fft_grid)
    r_len_i = int(r_len)

    fx = jnp.arange(nx, dtype=jnp.float64) / nx
    fy = jnp.arange(ny, dtype=jnp.float64) / ny
    fz = jnp.arange(nz, dtype=jnp.float64) / nz
    scale = jnp.complex128(int(sign) * 2j * jnp.pi)
    px = jnp.exp(scale * kvecs_frac[:, 0:1] * fx[None, :])    # (n_k, nx)
    py = jnp.exp(scale * kvecs_frac[:, 1:2] * fy[None, :])
    pz = jnp.exp(scale * kvecs_frac[:, 2:3] * fz[None, :])

    # Decode r_flat → (rx, ry, rz) on the slab.  Works for both Python-
    # int ``r0`` (broadcast as a constant) and jax-scalar ``r0`` (each
    # element a traced add).  ``nyn nz`` are static so divmod constants
    # fold cleanly.
    flat = r0 + jnp.arange(r_len_i, dtype=jnp.int32)         # (r_len,)
    rx = flat // (ny * nz)
    ry = (flat // nz) % ny
    rz = flat % nz

    # Per-slab-cell phase factor (n_k, r_len).
    p_x_slab = jnp.take(px, rx, axis=1)                      # (n_k, r_len)
    p_y_slab = jnp.take(py, ry, axis=1)
    p_z_slab = jnp.take(pz, rz, axis=1)
    phase_slab = p_x_slab * p_y_slab * p_z_slab              # (n_k, r_len)

    # Broadcast against slab's trailing r-axis; pad k with intermediate
    # broadcast axes (band, spinor, μ, ...) just like apply_bloch_phase.
    n_mid = slab.ndim - 2                                    # k + mid + r
    mid_shape = (1,) * n_mid
    phase_slab = phase_slab.reshape(phase_slab.shape[0], *mid_shape, r_len_i)
    return slab * phase_slab
