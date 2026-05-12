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
        if kvecs_frac is None:
            @jax.jit
            def fn(psi_, g_index_):
                box = _box_kernel(psi_, g_index_, ngkmax=ngkmax)
                out = jnp.fft.ifftn(box, axes=(-3, -2, -1), norm=norm)
                return _maybe_constrain(out, out_sharding)
            _RBOX_KERNEL_CACHE[key] = fn
        else:
            @jax.jit
            def fn(psi_, g_index_, kvecs_):
                box = _box_kernel(psi_, g_index_, ngkmax=ngkmax)
                rb = jnp.fft.ifftn(box, axes=(-3, -2, -1), norm=norm)
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
        if kvecs_frac is None:
            @jax.jit
            def fn(psi_, g_index_, r_mu_):
                box = _box_kernel(psi_, g_index_, ngkmax=ngkmax)
                rb = jnp.fft.ifftn(box, axes=(-3, -2, -1), norm=norm)
                out = rb[:, :, :, r_mu_[:, 0], r_mu_[:, 1], r_mu_[:, 2]]
                return _maybe_constrain(out, out_sharding)
            _RMU_KERNEL_CACHE[key] = fn
        else:
            @jax.jit
            def fn(psi_, g_index_, r_mu_, kvecs_):
                box = _box_kernel(psi_, g_index_, ngkmax=ngkmax)
                rb = jnp.fft.ifftn(box, axes=(-3, -2, -1), norm=norm)
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
        if kvecs_frac is None:
            @jax.jit
            def fn(psi_, g_index_, r0_):
                box = _box_kernel(psi_, g_index_, ngkmax=ngkmax)
                rb = jnp.fft.ifftn(box, axes=(-3, -2, -1), norm=norm)
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
                box = _box_kernel(psi_, g_index_, ngkmax=ngkmax)
                rb = jnp.fft.ifftn(box, axes=(-3, -2, -1), norm=norm)
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
        Split the leading ``n_q × n_rmu`` (collapsed) batch axis into
        this many sub-chunks before forming the FFT box.  Default 1
        materialises the full ``(n_q, n_rmu, nx, ny, nz)`` FFT box in
        memory — fine for MoS2 (~few GB) but **OOMs at CrI3 scale**
        (17.5 GB per intermediate × 4-5 live copies during the
        forward FFT + reshape + gather chain).  Setting
        ``fft_batch_chunks = n_rchunks`` (or higher) caps the working
        set to one rchunk's worth of memory — by construction known
        to fit, since the upstream r-chunk loop produces exactly that
        much data per iteration.  Implemented via ``jax.lax.scan``
        over the batch axis with the gather + accumulate fused
        per-chunk so the full FFT box is never materialised.

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
        sphere_id = id(sphere_arr.tobytes())

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
        # Static sphere_idx: bake into closure to keep it out of the
        # arg tuple (and to keep dtype consistent with the gather).
        if sphere_idx is not None:
            sphere_const = jnp.asarray(sphere_arr, dtype=jnp.int32)
        else:
            sphere_const = None

        def _gather_sphere(G_flat):
            """Per-q OR shared gather on the trailing axis of G_flat."""
            if sphere_const is None:
                return G_flat
            if not sphere_per_q:
                return jnp.take(G_flat, sphere_const, axis=-1)
            # Per-q gather: sphere_const has shape (n_q, ngkmax) and
            # G_flat has shape (n_q, *mid, n_rtot).  Broadcast the
            # sphere across any mid axes for take_along_axis.
            n_mid_axes = G_flat.ndim - 2
            sphere_b = sphere_const.reshape(
                sphere_const.shape[0],
                *((1,) * n_mid_axes),
                sphere_const.shape[1],
            )
            return jnp.take_along_axis(
                G_flat, sphere_b, axis=-1, mode='promise_in_bounds')

        # Capture the input sharding so intermediates inherit it (the
        # μ-sharded inputs land at ``P(*, ('x','y'), *)`` and JAX zero-
        # init / reshape / fftn don't always preserve the metadata).
        # Without this, CrI3-scale runs blow up to ~100 GB on a single
        # device from unsharded ``pad_buf`` / ``box`` intermediates.
        _in_sh = getattr(rchunk, "sharding", None)
        if _in_sh is not None and hasattr(_in_sh, "mesh"):
            _mesh = _in_sh.mesh
            _spec = tuple(getattr(_in_sh, "spec", ()))
            # μ axis (axis -2 of the rchunk) is sharded; spatial axis
            # -1 should not be.  Build the 3-D (q, μ, r) and 5-D
            # (q, μ, nx, ny, nz) variants explicitly.  Use ``None``s
            # to match rchunk.ndim positions.
            _mu_spec = _spec[-2] if len(_spec) >= 2 else None
            _sh3d = NamedSharding(
                _mesh, P(*((None,) * (len(_spec) - 2)), _mu_spec, None))
            _sh5d = NamedSharding(
                _mesh,
                P(*((None,) * (len(_spec) - 2)), _mu_spec, None, None, None))
        else:
            _sh3d = None
            _sh5d = None

        def _shard3(x):
            return (x if _sh3d is None
                    else jax.lax.with_sharding_constraint(x, _sh3d))

        def _shard5(x):
            return (x if _sh5d is None
                    else jax.lax.with_sharding_constraint(x, _sh5d))

        # Chunked path: fold the leading n_q axis into ``n_batch_chunks``
        # sub-chunks of size ``q_chunk = n_q // n_batch_chunks``.  Each
        # sub-chunk's FFT box is (q_chunk, μ, nx, ny, nz) — 1/n_chunks
        # the working set of the one-shot path.  For CrI3 J_3x3 (n_q=9,
        # μ=1504, FFT 45×45×120), one-shot's 17.5 GB FFT box → 5.9 GB
        # per chunk at n_batch_chunks=9, fits on 80 GB devices once
        # μ-sharded.  ``jax.lax.scan`` compiles the chunk body once
        # and reuses it.
        _n_q_kernel = int(rchunk.shape[0])
        if n_batch_chunks > 1 and _n_q_kernel % n_batch_chunks != 0:
            raise ValueError(
                f"accumulate_rchunk_to_gflat: fft_batch_chunks="
                f"{n_batch_chunks} must divide n_q={_n_q_kernel}.")
        _q_chunk = _n_q_kernel // n_batch_chunks

        def _per_chunk_contrib(rch_phased, sphere_slice_or_none):
            """Pad → FFT → sphere gather for one (q_chunk, ..., r_len) sub-batch."""
            pad_buf = jnp.zeros(
                (*rch_phased.shape[:-1], n_rtot), dtype=rch_phased.dtype)
            pad_buf = _shard3(pad_buf)
            pad_buf = jax.lax.dynamic_update_slice_in_dim(
                pad_buf, rch_phased, 0, axis=-1)
            # NB: ``r0_`` is folded into the phase before this point,
            # so the FFT input slab always starts at flat index 0 in
            # the per-chunk pad buffer.  This is equivalent to placing
            # the slab at ``r0_`` and multiplying the FFT output by
            # ``exp(-2πi G·r0)`` — but XLA's scan body needs a static
            # start to keep the dynamic_update_slice cheap.  We use
            # the phase-fold variant below for the kvecs path, and the
            # no-qvec path is just zero-r0 by construction.
            box = pad_buf.reshape(*rch_phased.shape[:-1], nx, ny, nz)
            box = _shard5(box)
            G_box = jnp.fft.fftn(box, axes=(-3, -2, -1), norm=norm)
            G_box = _shard5(G_box)
            G_flat = G_box.reshape(*rch_phased.shape[:-1], n_rtot)
            G_flat = _shard3(G_flat)
            if sphere_const is None:
                return G_flat
            if not sphere_per_q:
                return jnp.take(G_flat, sphere_const, axis=-1)
            n_mid = G_flat.ndim - 2
            sphere_b = sphere_slice_or_none.reshape(
                sphere_slice_or_none.shape[0],
                *((1,) * n_mid),
                sphere_slice_or_none.shape[1],
            )
            return jnp.take_along_axis(
                G_flat, sphere_b, axis=-1, mode='promise_in_bounds')

        if qvec_frac is None:
            @partial(jax.jit, donate_argnums=(1,))
            def fn(rch_, acc_, r0_):
                # One-shot path when n_batch_chunks==1; chunk over n_q
                # otherwise.  Note: the no-qvec path doesn't fold r0
                # into a phase (no phase at all), so the FFT slab must
                # be placed at the correct r0 in the pad buffer — we
                # do that inside the chunk body using ``r0_``.
                if n_batch_chunks == 1:
                    pad_buf = jnp.zeros(
                        (*rch_.shape[:-1], n_rtot), dtype=rch_.dtype)
                    pad_buf = _shard3(pad_buf)
                    pad_buf = jax.lax.dynamic_update_slice_in_dim(
                        pad_buf, rch_, r0_, axis=-1)
                    pad_buf = _shard3(pad_buf)
                    box = pad_buf.reshape(*rch_.shape[:-1], nx, ny, nz)
                    box = _shard5(box)
                    G_box = jnp.fft.fftn(box, axes=(-3, -2, -1), norm=norm)
                    G_box = _shard5(G_box)
                    G_flat = G_box.reshape(*rch_.shape[:-1], n_rtot)
                    G_flat = _shard3(G_flat)
                    return acc_ + _gather_sphere(G_flat)

                def body(acc, i):
                    start = i * _q_chunk
                    rch_chunk = jax.lax.dynamic_slice_in_dim(
                        rch_, start, _q_chunk, axis=0)
                    pad_buf = jnp.zeros(
                        (_q_chunk, *rch_.shape[1:-1], n_rtot),
                        dtype=rch_.dtype)
                    pad_buf = jax.lax.dynamic_update_slice_in_dim(
                        pad_buf, rch_chunk, r0_, axis=-1)
                    box = pad_buf.reshape(
                        _q_chunk, *rch_.shape[1:-1], nx, ny, nz)
                    G_box = jnp.fft.fftn(
                        box, axes=(-3, -2, -1), norm=norm)
                    G_flat = G_box.reshape(
                        _q_chunk, *rch_.shape[1:-1], n_rtot)
                    if sphere_const is None:
                        contrib = G_flat
                    elif not sphere_per_q:
                        contrib = jnp.take(G_flat, sphere_const, axis=-1)
                    else:
                        sphere_chunk = jax.lax.dynamic_slice_in_dim(
                            sphere_const, start, _q_chunk, axis=0)
                        n_mid = G_flat.ndim - 2
                        sphere_b = sphere_chunk.reshape(
                            _q_chunk, *((1,) * n_mid),
                            sphere_chunk.shape[1])
                        contrib = jnp.take_along_axis(
                            G_flat, sphere_b, axis=-1,
                            mode='promise_in_bounds')
                    acc_chunk = jax.lax.dynamic_slice_in_dim(
                        acc, start, _q_chunk, axis=0)
                    acc = jax.lax.dynamic_update_slice_in_dim(
                        acc, acc_chunk + contrib, start, axis=0)
                    return acc, None
                acc_, _ = jax.lax.scan(
                    body, acc_, jnp.arange(n_batch_chunks, dtype=jnp.int32))
                return acc_
            _RCHUNK_TO_GFLAT_CACHE[key] = fn
        else:
            @partial(jax.jit, donate_argnums=(1,))
            def fn(rch_, acc_, r0_, qvec_):
                if n_batch_chunks == 1:
                    rch_phased = apply_bloch_phase_on_slice(
                        rch_, qvec_, fft_grid_t, r0_, r_len_i, sign=-1)
                    rch_phased = _shard3(rch_phased)
                    pad_buf = jnp.zeros(
                        (*rch_.shape[:-1], n_rtot), dtype=rch_.dtype)
                    pad_buf = _shard3(pad_buf)
                    pad_buf = jax.lax.dynamic_update_slice_in_dim(
                        pad_buf, rch_phased, r0_, axis=-1)
                    pad_buf = _shard3(pad_buf)
                    box = pad_buf.reshape(*rch_.shape[:-1], nx, ny, nz)
                    box = _shard5(box)
                    G_box = jnp.fft.fftn(box, axes=(-3, -2, -1), norm=norm)
                    G_box = _shard5(G_box)
                    G_flat = G_box.reshape(*rch_.shape[:-1], n_rtot)
                    G_flat = _shard3(G_flat)
                    return acc_ + _gather_sphere(G_flat)

                def body(acc, i):
                    start = i * _q_chunk
                    rch_chunk = jax.lax.dynamic_slice_in_dim(
                        rch_, start, _q_chunk, axis=0)
                    qvec_chunk = jax.lax.dynamic_slice_in_dim(
                        qvec_, start, _q_chunk, axis=0)
                    rch_phased = apply_bloch_phase_on_slice(
                        rch_chunk, qvec_chunk, fft_grid_t, r0_, r_len_i,
                        sign=-1)
                    pad_buf = jnp.zeros(
                        (_q_chunk, *rch_.shape[1:-1], n_rtot),
                        dtype=rch_.dtype)
                    pad_buf = jax.lax.dynamic_update_slice_in_dim(
                        pad_buf, rch_phased, r0_, axis=-1)
                    box = pad_buf.reshape(
                        _q_chunk, *rch_.shape[1:-1], nx, ny, nz)
                    G_box = jnp.fft.fftn(
                        box, axes=(-3, -2, -1), norm=norm)
                    G_flat = G_box.reshape(
                        _q_chunk, *rch_.shape[1:-1], n_rtot)
                    if sphere_const is None:
                        contrib = G_flat
                    elif not sphere_per_q:
                        contrib = jnp.take(G_flat, sphere_const, axis=-1)
                    else:
                        sphere_chunk = jax.lax.dynamic_slice_in_dim(
                            sphere_const, start, _q_chunk, axis=0)
                        n_mid = G_flat.ndim - 2
                        sphere_b = sphere_chunk.reshape(
                            _q_chunk, *((1,) * n_mid),
                            sphere_chunk.shape[1])
                        contrib = jnp.take_along_axis(
                            G_flat, sphere_b, axis=-1,
                            mode='promise_in_bounds')
                    acc_chunk = jax.lax.dynamic_slice_in_dim(
                        acc, start, _q_chunk, axis=0)
                    acc = jax.lax.dynamic_update_slice_in_dim(
                        acc, acc_chunk + contrib, start, axis=0)
                    return acc, None
                acc_, _ = jax.lax.scan(
                    body, acc_, jnp.arange(n_batch_chunks, dtype=jnp.int32))
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
