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
    "apply_bloch_phase", "apply_bloch_phase_on_slice",
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

def _local_box_fft(psi: jax.Array, *, kind: str, norm: str,
                   mesh: Mesh | None = None):
    """Sharded local FFT for the box ``psi.shape[:-1] + (nx, ny, nz)``.

    Returns a callable ``f(box) -> (i)fftn(box, axes=(-3, -2, -1))`` whose
    output sharding preserves psi's leading layout (FFT axes replicated).
    ``kind`` is ``'fftn'`` or ``'ifftn'``.

    Mesh resolution order: explicit ``mesh=`` arg → ``psi.sharding.mesh``
    → trivial 1×1 mesh (test-bench fallback).  Callers operating inside
    another jit (where ``psi`` is a tracer whose sharding doesn't carry
    mesh) must pass ``mesh=`` explicitly.
    """
    from common.fft_helpers import (
        make_sharded_ifftn_3d, make_sharded_fftn_3d)
    sh = getattr(psi, "sharding", None)
    sh_mesh = getattr(sh, "mesh", None) if sh is not None else None
    m = mesh or sh_mesh or Mesh(
        np.asarray(jax.devices()[:1]).reshape(1, 1), axis_names=('x', 'y'))
    # Spec: preserve psi's leading layout (if it has one) and replicate
    # the three appended FFT axes.  Inputs without a real sharding fall
    # back to fully replicated.
    leading = (tuple(sh.spec)[:-1]
               if sh is not None and hasattr(sh, "spec") and sh_mesh is not None
               else (None,) * (psi.ndim - 1))
    spec = P(*leading, None, None, None)
    factory = (make_sharded_ifftn_3d if kind == 'ifftn'
               else make_sharded_fftn_3d)
    return factory(m, spec, spec, norm=norm, axes=(-3, -2, -1))


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
    mesh: Mesh | None = None,
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
           kvecs_shape, _sharding_key(psi), out_sharding)
    fn = _RBOX_KERNEL_CACHE.get(key)
    if fn is None:
        ifftn = _local_box_fft(psi, kind='ifftn', norm=norm, mesh=mesh)
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
    norm: str = "backward",
    kvecs_frac: np.ndarray | jax.Array | None = None,
    mesh: Mesh | None = None,
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
    out_sharding = _output_sharding(psi, n_extra_axes=1)
    key = (psi.shape, tuple(g_index.shape), ngkmax, fft_grid_t, n_rmu,
           norm, kvecs_shape, _sharding_key(psi), out_sharding)
    fn = _RMU_KERNEL_CACHE.get(key)
    if fn is None:
        ifftn = _local_box_fft(psi, kind='ifftn', norm=norm, mesh=mesh)
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


def to_rchunk(
    psi: jax.Array,
    g_index: np.ndarray | jax.Array,
    fft_grid: Sequence[int],
    r0,
    r_len: int,
    *,
    norm: str = "backward",
    kvecs_frac: np.ndarray | jax.Array | None = None,
    mesh: Mesh | None = None,
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
           norm, kvecs_shape, _sharding_key(psi), out_sharding)
    fn = _RCHUNK_KERNEL_CACHE.get(key)
    if fn is None:
        ifftn = _local_box_fft(psi, kind='ifftn', norm=norm, mesh=mesh)
        if kvecs_frac is None:
            @jax.jit
            def fn(psi_, g_index_, r0_):
                rb = ifftn(_box_kernel(psi_, g_index_, ngkmax=ngkmax))
                rb_flat = rb.reshape(*rb.shape[:3], n_rtot)
                out = jax.lax.dynamic_slice_in_dim(rb_flat, r0_, r_len_i, axis=-1)
                return _maybe_constrain(out, out_sharding)
            _RCHUNK_KERNEL_CACHE[key] = fn
        else:
            @jax.jit
            def fn(psi_, g_index_, r0_, kvecs_):
                # Phase-after-slice: IFFT → flatten → slice → phase on
                # the r_len-cell slab (not on the full nx·ny·nz box).
                # Mathematically equivalent (multiplication commutes
                # with slicing along r); cuts the per-r phase work from
                # n_k·nx·ny·nz to n_k·r_len.
                rb = ifftn(_box_kernel(psi_, g_index_, ngkmax=ngkmax))
                rb_flat = rb.reshape(*rb.shape[:3], n_rtot)
                slab = jax.lax.dynamic_slice_in_dim(rb_flat, r0_, r_len_i, axis=-1)
                slab = apply_bloch_phase_on_slice(
                    slab, kvecs_, fft_grid_t, r0_, r_len_i)
                return _maybe_constrain(slab, out_sharding)
            _RCHUNK_KERNEL_CACHE[key] = fn
    g_index_j = jnp.asarray(g_index, dtype=jnp.int32)
    r0_arg = (jnp.int32(int(r0)) if isinstance(r0, (int, np.integer)) else r0)
    if kvecs_frac is None:
        return fn(psi, g_index_j, r0_arg)
    return fn(psi, g_index_j, r0_arg,
              jnp.asarray(kvecs_frac, dtype=jnp.float64))


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
    fft_grid: Sequence[int],
    r0,
    sphere_idx: np.ndarray | jax.Array | None,
    qvec_frac: np.ndarray | jax.Array | None = None,
    norm: str = "backward",
    fft_batch_chunks: int = 1,
    mesh: Mesh | None = None,
) -> jax.Array:
    """Add ``FFT(pad(phase(rchunk)))[sphere_idx]`` into ``gflat_acc``.

    Parameters
    ----------
    rchunk
        Trailing shape ``(n_q, ..., r_len)`` c128.  Leading axis is
        the q-axis used to look up the per-q Bloch phase.
        Intermediate axes (e.g. μ, spinor) are broadcast through the
        FFT and the phase.
    gflat_acc
        Trailing shape ``(n_q, ..., n_G_sph)`` c128.  Same leading
        axes as ``rchunk``; trailing axis is the G-sphere subset.
        Donated to the inner jit — its buffer is reused in place.
    fft_grid
        Static ``(nx, ny, nz)``.  Defines the flat-r enumeration
        (``r = rx·ny·nz + ry·nz + rz``) and the FFT shape.
    r0
        Python int or jax-scalar — flat-r start of the slab in
        ``[0, nx·ny·nz)``.
    sphere_idx
        Static int32 flat-FFT indices to gather the G-sphere subset.
        Two shapes:

        * ``(n_G_sph,)`` — a **shared** sphere applied to every q.
          Same gather index for all q; cheap ``jnp.take`` on the
          trailing axis.
        * ``(n_q, ngkmax)`` — a **per-q** sphere (WFN.h5-style padded
          layout).  ``n_q`` must match ``rchunk.shape[0]`` and
          ``gflat_acc.shape[0]``.  Each q gets its own gather row;
          pad slots within a row are sentinel flat-FFT indices whose
          coeffs will be zeroed by the caller post-loop.  Compiles
          to ``take_along_axis`` with ``mode='promise_in_bounds'``
          to keep XLA's x64+shard_map verifier happy (sphere_idx is
          guaranteed in-bounds by construction).

        ``None`` keeps the whole flat-FFT axis (``n_G_sph = nx·ny·nz``).
    qvec_frac
        Optional ``(n_q, 3)`` fractional q-vectors.  When given, the
        pre-FFT slab is multiplied by ``exp(-2πi q·r)`` on the slab
        (matches the convention used by V_q's
        ``_zeta_disk_to_G``).  ``None`` skips the phase.
    norm
        Forwarded to :func:`jnp.fft.fftn`.
    fft_batch_chunks
        Split the μ axis (axis 1) of ``rchunk`` into this many
        sub-chunks before forming the FFT box.  Default 1
        materialises the full ``(n_q, n_rmu_padded, nx, ny, nz)`` FFT
        box in memory — fine for MoS2 (~few GB) but **OOMs at CrI3
        scale** (17.5 GB per intermediate × 4-5 live copies during
        the forward FFT + reshape + gather chain).  Setting
        ``fft_batch_chunks = n_rchunks`` (or higher) caps the working
        set to one rchunk's worth of memory — by construction known
        to fit, since the upstream r-chunk loop produces exactly that
        much data per iteration.  Chunking on the μ axis (rather than
        n_q) handles the n_q=1 case (Γ-only debug runs) and aligns
        with the already-μ-sharded sharding contract so each
        per-chunk pad_buf / box / G_box is naturally decomposed
        across ranks by the ``with_sharding_constraint`` calls.
        Must divide ``n_rmu_padded``.  Implemented via
        ``jax.lax.scan`` over the chunk axis with the gather +
        accumulate fused per-chunk so the full FFT box is never
        materialised.

    Returns
    -------
    Updated ``gflat_acc`` (same shape, same sharding).  Donation makes
    the update in-place under jit; outside-jit callers should rebind.
    """
    fft_grid_t = tuple(int(s) for s in fft_grid)
    nx, ny, nz = fft_grid_t
    n_rtot = nx * ny * nz
    r_len_i = int(rchunk.shape[-1])
    if isinstance(r0, (int, np.integer)):
        if r0 < 0 or int(r0) + r_len_i > n_rtot:
            raise ValueError(
                f"accumulate_rchunk_to_gflat: r-slab "
                f"[{int(r0)}, {int(r0) + r_len_i}) out of [0, {n_rtot})")
    if sphere_idx is None:
        n_G_sph = n_rtot
        sphere_id = None
        sphere_per_q = False
        sphere_arr = None
    else:
        sphere_arr = np.asarray(sphere_idx, dtype=np.int32)
        if sphere_arr.ndim == 1:
            n_G_sph = int(sphere_arr.shape[0])
            sphere_per_q = False
        elif sphere_arr.ndim == 2:
            n_G_sph = int(sphere_arr.shape[1])
            sphere_per_q = True
            if int(sphere_arr.shape[0]) != int(rchunk.shape[0]):
                raise ValueError(
                    f"accumulate_rchunk_to_gflat: per-q sphere_idx has "
                    f"shape {sphere_arr.shape} but rchunk has "
                    f"n_q={int(rchunk.shape[0])}.")
        else:
            raise ValueError(
                f"accumulate_rchunk_to_gflat: sphere_idx must be 1-D "
                f"(shared) or 2-D (per-q); got shape {sphere_arr.shape}.")
        # Content-based hash — id() of the temporary tobytes() can be
        # reused after the temp is GC'd, producing cache collisions.
        sphere_id = hash(sphere_arr.tobytes())

    qvec_shape = (None if qvec_frac is None
                  else tuple(int(s) for s in np.shape(qvec_frac)))

    n_batch_chunks = max(1, int(fft_batch_chunks))
    key = (
        tuple(int(s) for s in rchunk.shape),
        tuple(int(s) for s in gflat_acc.shape),
        fft_grid_t, r_len_i, n_G_sph, sphere_id, sphere_per_q,
        norm, qvec_shape, n_batch_chunks,
        _sharding_key(rchunk), _sharding_key(gflat_acc),
    )
    fn = _RCHUNK_TO_GFLAT_CACHE.get(key)
    if fn is None:
        # Static closure: per-q or shared sphere index baked in.
        sphere_const = (jnp.asarray(sphere_arr, dtype=jnp.int32)
                        if sphere_idx is not None else None)

        def _gather_sphere(G_flat):
            """Per-q or shared gather on the trailing axis of G_flat.

            Shared sphere (1-D, ngkmax,) → ``jnp.take``.  Per-q sphere
            (2-D, n_q × ngkmax) → ``take_along_axis`` with the sphere
            broadcast across the middle (μ, ...) axes of G_flat.
            """
            if sphere_const is None:
                return G_flat
            if not sphere_per_q:
                return jnp.take(G_flat, sphere_const, axis=-1)
            n_mid_axes = G_flat.ndim - 2
            sphere_b = sphere_const.reshape(
                sphere_const.shape[0],
                *((1,) * n_mid_axes),
                sphere_const.shape[1],
            )
            return jnp.take_along_axis(
                G_flat, sphere_b, axis=-1, mode='promise_in_bounds')

        # Per-chunk intermediates need ``with_sharding_constraint`` so
        # ``jnp.zeros`` / reshape inherit the μ-shard — otherwise XLA
        # materialises replicated tensors (~100 GB single-device at CrI3
        # scale).  The FFT itself goes through ``_local_box_fft`` which
        # uses ``make_sharded_*fftn_3d`` directly.
        _in_sh = getattr(rchunk, "sharding", None)
        if (_in_sh is not None
                and getattr(_in_sh, "mesh", None) is not None
                and len(getattr(_in_sh, "spec", ())) >= 2):
            _mesh = _in_sh.mesh
            _spec = tuple(_in_sh.spec)
            _leading = _spec[:-2]  # everything before the μ axis
            _mu_spec = _spec[-2]
            _p_prod = int(np.prod([_mesh.shape[a] for a in _mesh.axis_names]))
            _sh_3d = NamedSharding(_mesh, P(*_leading, _mu_spec, None))
            _sh_5d = NamedSharding(
                _mesh, P(*_leading, _mu_spec, None, None, None))
        else:
            _sh_3d = _sh_5d = None
            _p_prod = 1

        def _shard3(x): return _maybe_constrain(x, _sh_3d)
        def _shard5(x): return _maybe_constrain(x, _sh_5d)

        local_fftn = _local_box_fft(rchunk, kind='fftn', norm=norm, mesh=mesh)

        # μ-axis chunking.  ``n_batch_chunks`` must divide
        # ``n_mu_local = n_mu_padded / p_prod`` so each chunk's
        # per-rank shard stays integer-sized.  Chunking on μ (not n_q)
        # handles the n_q=1 case (Γ-only debug runs) and aligns with
        # the μ-sharded layout so each per-chunk pad_buf / box / G_box
        # is naturally decomposed across ranks by ``_shard3`` /
        # ``_shard5`` below.  ``n_batch_chunks=1`` is a length-1 scan
        # that XLA's loop optimiser folds away — same emitted HLO as
        # an inline body, with no extra trace work.
        _n_mu_padded = int(rchunk.shape[1])
        _n_mu_local = _n_mu_padded // _p_prod
        if n_batch_chunks > 1 and _n_mu_local % n_batch_chunks != 0:
            raise ValueError(
                f"accumulate_rchunk_to_gflat: fft_batch_chunks="
                f"{n_batch_chunks} must divide n_mu_local="
                f"{_n_mu_local} (= n_mu_padded/{_p_prod}).")
        _mu_chunk = _n_mu_padded // n_batch_chunks

        # Single scan body for every path: phase (if qvec_) → pad at
        # r0 → FFT box → forward FFT → flatten → sphere gather →
        # accumulate into the corresponding μ slice of ``acc``.  The
        # ``qvec_ is None`` branch folds away at trace time.
        def _chunk_body(acc, i, rch_, r0_, qvec_):
            start = i * _mu_chunk
            rch_chunk = _shard3(jax.lax.dynamic_slice_in_dim(
                rch_, start, _mu_chunk, axis=1))
            if qvec_ is not None:
                rch_chunk = _shard3(apply_bloch_phase_on_slice(
                    rch_chunk, qvec_, fft_grid_t, r0_, r_len_i, sign=-1))
            box_shape = (rch_.shape[0], _mu_chunk, *rch_.shape[2:-1])
            pad_buf = _shard3(jnp.zeros(
                (*box_shape, n_rtot), dtype=rch_.dtype))
            pad_buf = _shard3(jax.lax.dynamic_update_slice_in_dim(
                pad_buf, rch_chunk, r0_, axis=-1))
            box = _shard5(pad_buf.reshape(*box_shape, nx, ny, nz))
            G_box = local_fftn(box)
            G_flat = _shard3(G_box.reshape(*box_shape, n_rtot))
            contrib = _gather_sphere(G_flat)
            acc_chunk = jax.lax.dynamic_slice_in_dim(
                acc, start, _mu_chunk, axis=1)
            return jax.lax.dynamic_update_slice_in_dim(
                acc, acc_chunk + contrib, start, axis=1), None

        # Two thin wrappers (one per qvec presence) because ``None``
        # can't be a traced jit arg; both fall through to the same
        # scan over ``_chunk_body``.
        _iters = jnp.arange(n_batch_chunks, dtype=jnp.int32)
        if qvec_frac is None:
            @partial(jax.jit, donate_argnums=(1,))
            def fn(rch_, acc_, r0_):
                acc_, _ = jax.lax.scan(
                    lambda a, i: _chunk_body(a, i, rch_, r0_, None),
                    acc_, _iters)
                return acc_
        else:
            @partial(jax.jit, donate_argnums=(1,))
            def fn(rch_, acc_, r0_, qvec_):
                acc_, _ = jax.lax.scan(
                    lambda a, i: _chunk_body(a, i, rch_, r0_, qvec_),
                    acc_, _iters)
                return acc_
        _RCHUNK_TO_GFLAT_CACHE[key] = fn

    r0_arg = (jnp.int32(int(r0)) if isinstance(r0, (int, np.integer)) else r0)
    if qvec_frac is None:
        return fn(rchunk, gflat_acc, r0_arg)
    return fn(rchunk, gflat_acc, r0_arg,
              jnp.asarray(qvec_frac, dtype=jnp.float64))


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
