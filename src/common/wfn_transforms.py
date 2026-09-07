"""Transforms from G-flat ψ to FFT-box / r-space / centroid / r-chunk; see docs/architecture/zeta_fit_face_psi_cct.md."""
from __future__ import annotations

import gc
from functools import partial
from typing import Sequence, TYPE_CHECKING

import jax
import jax.numpy as jnp
import numpy as np
from runtime.padding import pad_axis, spec_divisor
from common.shard_map import shard_map
from common.staged_reshard import band_to_product_r_reshard
from common.wfn_layout import band_sphere_spec
from common.gpu_utils import worst_process_resident_bytes
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P
from common.fft_helpers import local_fftn3, local_ifftn3

if TYPE_CHECKING:
    from common.meta import Meta
    from runtime.aot_memory import AotPeakBreakdown


__all__ = [
    "FULL_BLOCH_TRANSFORM_SCHEME",
    "to_box", "to_rbox", "to_rmu", "to_rchunk",
    "to_rmu_inner", "to_rchunk_inner", "to_rpoints_inner",
    "take_rchunk_padded",
    "gflat_to_rchunk_aot_memory", "gflat_to_rchunk_aot_peak_bytes",
    "apply_bloch_phase", "apply_bloch_phase_on_slice", "apply_bloch_phase_at",
    "gflat_to_rmu",
    "accumulate_rchunk_to_gflat",
    # ψ(G)-loading helpers relocated from the former common/load_wfns.py.
    # These compose WfnLoader.load with the transforms above; they take a
    # WfnLoader as their first ``wfn`` argument.
    "get_enk_bandrange",
    "load_kpoint_fftbox",
    "load_kpoint_fftbox_local",
    "process_local_mesh",
    "read_Gvecs_to_devices",
    "prepare_rchunk_carrier",
    "iter_psi_rchunk_bandwise",
    "load_centroids_band_chunked",
    "load_psi_gflat_padded",
]


#: Versioned physics convention for the full-Bloch real-space rows produced
#: by this owner and :class:`common.psi_G_store.PsiGStore`.  Persisted
#: consumers must stamp this word: a WFN content fingerprint alone cannot
#: distinguish coefficients transformed with the old independently rebuilt
#: modular k table from coefficients transformed with loader-paired k/G
#: representatives.
FULL_BLOCH_TRANSFORM_SCHEME = (
    "lorrax-full-bloch-v1:loader-paired-k-g:ifftn-ortho:phase-plus")


# ---------------------------------------------------------------------------
# Kernel cache — one signature-keyed cache for every jit factory in this
# module.  Keys are ``(kernel_name, *signature_tuple)`` so different
# factories never collide.  Replaces the per-factory ``_X_CACHE`` dicts
# (and inline ``if fn is None: ... cache[key] = fn`` blocks) with one
# central ``_cached_jit`` helper.
# ---------------------------------------------------------------------------

_KERNEL_CACHE: dict = {}
_RCHUNK_PEAK_CACHE: dict = {}


def _cached_jit(name: str, key: tuple, build):
    """Return ``_KERNEL_CACHE[(name, *key)]`` or build + cache it."""
    full_key = (name, *key)
    fn = _KERNEL_CACHE.get(full_key)
    if fn is None:
        fn = build()
        _KERNEL_CACHE[full_key] = fn
    return fn


# Module-level dedup for the device-resident ``(nk, nx, ny, nz) int32``
# g_index buffer captured by ``build()`` closures.  Without this cache,
# every distinct ``_cached_jit`` key would otherwise build a NEW compiled fn
# with a NEW captured ``jnp.asarray(g_arr, dtype=jnp.int32)`` REPLICATED buffer
# — leaking +1 buffer per channel from the centroid-load path on top of
# the (now-fixed) ``psi_G_store._g_index_dev`` leak (agent_h §3 Finding
# 3).  Keyed by content-hash of the numpy g_index so different k-sets
# (full_bz vs ibz) get distinct buffers but identical numpy bytes share.
_GINDEX_DEV_CACHE: dict = {}

# Identity fast path in FRONT of the content hash.  ``hash(a.tobytes())``
# copies and digests the whole table: MEASURED 152 ms for the 0.16 GB
# int32 g_index of a production full-BZ load (0.062 GB / 54 ms at MoS2
# 3x3 scale), paid on EVERY call.  In practice the caller hands back the
# very same object each time — ``WfnLoader.box_index`` memoises it — so
# an ``id()`` probe answers first.  ``id()`` alone is unsafe (a freed
# array's address gets recycled), hence the weakref: if the array we
# recorded has died, the entry is stale and we fall through to the hash.
# Values are (weakref-to-numpy, device buffer); we deliberately do NOT
# hold a strong reference to the numpy array — that would keep a
# 0.16 GB host table alive past its loader.
_GINDEX_DEV_BY_ID: dict = {}


def _cached_gindex_dev(g_arr) -> "jax.Array":
    """Cache the ``jnp.asarray(g_arr, dtype=jnp.int32)`` REPLICATED device buffer by content hash; see docs/architecture/zeta_fit_face_psi_cct.md."""
    if isinstance(g_arr, jax.Array):
        # Already a device buffer.  ``jnp.asarray`` is identity when
        # dtype already matches; explicit short-circuit makes the
        # canonical-buffer path obvious to readers.
        if g_arr.dtype == jnp.int32:
            return g_arr
        return jnp.asarray(g_arr, dtype=jnp.int32)
    # Identity probe first — see ``_GINDEX_DEV_BY_ID``.  A hit means this
    # is literally the object we hashed before, so the content cannot
    # have changed identity underneath us; a miss (or a dead weakref)
    # costs one dict lookup and falls through to the content hash, which
    # still deduplicates distinct objects with identical bytes.
    import weakref
    ident = id(g_arr)
    seen = _GINDEX_DEV_BY_ID.get(ident)
    if seen is not None:
        ref, dev = seen
        if ref() is g_arr:
            return dev
        del _GINDEX_DEV_BY_ID[ident]          # address recycled — stale

    key = ('g_index_dev', hash(g_arr.tobytes()),
           tuple(int(s) for s in g_arr.shape))
    hit = _GINDEX_DEV_CACHE.get(key)
    if hit is not None:
        dev = hit
    else:
        dev = jnp.asarray(g_arr, dtype=jnp.int32)
        _GINDEX_DEV_CACHE[key] = dev
    try:
        _GINDEX_DEV_BY_ID[ident] = (weakref.ref(g_arr), dev)
    except TypeError:                          # not weak-referenceable
        pass
    return dev


def _resolve_gindex_dev(g_index):
    """Return ``(g_index_dev_jax, cache_id)`` for either a numpy array or a ``jax.Array`` g_index — without a device→host roundtrip when the caller has the canonical buffer in hand; see docs/architecture/zeta_fit_face_psi_cct.md."""
    if isinstance(g_index, jax.Array):
        if g_index.dtype != jnp.int32:
            # Edge case: caller's canonical buffer is non-int32.  Cast
            # is cheap if dtype already matches (jnp.asarray returns
            # identity), only fires when caller built a non-canonical
            # buffer manually.  cache_id still keys on the source
            # jax.Array's id to keep the canonical-buffer fast path.
            g_index_dev = jnp.asarray(g_index, dtype=jnp.int32)
        else:
            g_index_dev = g_index
        shape = tuple(int(s) for s in g_index.shape)
        return g_index_dev, ('jax_id', id(g_index), shape)
    g_arr = np.asarray(g_index, dtype=np.int32)
    g_index_dev = _cached_gindex_dev(g_arr)
    shape = tuple(int(s) for s in g_arr.shape)
    return g_index_dev, ('np_hash', hash(g_arr.tobytes()), shape)


# ---------------------------------------------------------------------------
# Shared kernel: G-flat ψ + g_index → FFT-box ψ (zero-sentinel gather)
# ---------------------------------------------------------------------------
#
# ``common.gvec_fft_box`` owns the host-built inverse index; this is the sole
# device gather that consumes it.  For each FFT-box cell (nx, ny, nz),
# ``g_index[k, nx, ny, nz]`` gives the position along the G-axis of psi
# to gather from; positions equal to ``ngkmax`` map to a synthetic
# zero slot appended on the G-axis before the gather.  One ``take``
# call fills the whole box; no per-k loop, no scatter, no per-cell
# masking.

def _box_kernel(psi: jax.Array, g_index: jax.Array, *, ngkmax: int) -> jax.Array:
    """psi: (n_k, nb, ns, ngkmax) c128 — band-sharded acceptable; see docs/architecture/zeta_fit_face_psi_cct.md."""
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
    """Partition spec for ``psi``, always of length ``psi.ndim``; see docs/architecture/zeta_fit_face_psi_cct.md."""
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
    """Sharded local FFT for the box ``psi.shape[:-1] + (nx, ny, nz)``; see docs/architecture/zeta_fit_face_psi_cct.md."""
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
    """Hashable signature of psi's sharding (mesh identity + spec); see docs/architecture/zeta_fit_face_psi_cct.md."""
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
    """Scatter G-flat ψ into the FFT box; see docs/architecture/zeta_fit_face_psi_cct.md."""
    ngkmax = int(psi.shape[-1])
    fft_grid_t = tuple(int(s) for s in fft_grid)
    out_sharding = _output_sharding(psi, mesh, n_extra_axes=3)
    key = (psi.shape, tuple(g_index.shape), ngkmax, fft_grid_t,
           _sharding_key(psi), out_sharding)

    def build():
        @jax.jit
        def fn(psi_, g_index_):
            out = _box_kernel(psi_, g_index_, ngkmax=ngkmax)
            return _maybe_constrain(out, out_sharding)
        return fn

    fn = _cached_jit('to_box', key, build)
    g_index_j = _cached_gindex_dev(g_index)
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
    """Scatter ψ → FFT box → IFFT to r-space (+ optional Bloch phase); see docs/architecture/zeta_fit_face_psi_cct.md."""
    ngkmax = int(psi.shape[-1])
    fft_grid_t = tuple(int(s) for s in fft_grid)
    kvecs_shape = (None if kvecs_frac is None
                   else tuple(int(s) for s in np.shape(kvecs_frac)))
    out_sharding = _output_sharding(psi, mesh, n_extra_axes=3)
    key = (psi.shape, tuple(g_index.shape), ngkmax, fft_grid_t, norm,
           kvecs_shape, _sharding_key(psi), out_sharding)

    def build():
        ifftn = _local_box_fft(psi, mesh, kind='ifftn', norm=norm)
        if kvecs_frac is None:
            @jax.jit
            def fn(psi_, g_index_):
                box = _box_kernel(psi_, g_index_, ngkmax=ngkmax)
                return _maybe_constrain(ifftn(box), out_sharding)
            return fn
        @jax.jit
        def fn(psi_, g_index_, kvecs_):
            box = _box_kernel(psi_, g_index_, ngkmax=ngkmax)
            rb = apply_bloch_phase(ifftn(box), kvecs_, fft_grid_t)
            return _maybe_constrain(rb, out_sharding)
        return fn

    fn = _cached_jit('to_rbox', key, build)
    g_index_j = _cached_gindex_dev(g_index)
    if kvecs_frac is None:
        return fn(psi, g_index_j)
    return fn(psi, g_index_j, jnp.asarray(kvecs_frac, dtype=jnp.float64))


def from_rbox(
    psi_r: jax.Array,
    gvecs: np.ndarray | jax.Array,
    *,
    mesh: Mesh,
    norm: str = "backward",
    g_mask: np.ndarray | jax.Array | None = None,
) -> jax.Array:
    """r-space FFT box → G-sphere; see docs/architecture/zeta_fit_face_psi_cct.md."""
    gv = np.asarray(gvecs)
    if gv.ndim != 2 or gv.shape[1] != 3:
        raise ValueError(
            f"from_rbox: gvecs must be (ngkmax, 3), got {gv.shape}")
    ngkmax = int(gv.shape[0])
    out_spec = _spec_of(psi_r)
    key = (psi_r.shape, ngkmax, norm, _sharding_key(psi_r),
           g_mask is not None)

    def build():
        # NOT ``_local_box_fft``: that helper derives the BOX spec from a
        # SPHERE-shaped ψ (``P(*_spec_of(psi)[:-1], None, None, None)``, i.e.
        # 4-D in → 6-entry spec), and here the input is ALREADY the box, so it
        # would emit an 8-entry spec and shard_map rejects it ("in_specs entry
        # too long").  Same factory, correct arity for a box operand — the
        # FFT axes replicated, only the leading (k, band, spinor) axes
        # sharded.
        from common.fft_helpers import make_sharded_fftn_3d
        box_spec = P(*out_spec[:3], None, None, None)
        fftn = make_sharded_fftn_3d(mesh, box_spec, box_spec,
                                    norm=norm, axes=(-3, -2, -1))
        sharding = NamedSharding(mesh, P(*out_spec[:3], None))

        @jax.jit
        def fn(psi_r_, gx_, gy_, gz_, mask_):
            box = fftn(psi_r_)
            # Gather the sphere out of the box.  Advanced indexing on the
            # last three (replicated) axes only — no cross-rank op, so the
            # band sharding rides through untouched.
            out = box[..., gx_, gy_, gz_]
            if mask_ is not None:
                out = out * mask_
            return _maybe_constrain(out, sharding)
        return fn

    fn = _cached_jit('from_rbox', key, build)
    gx = jnp.asarray(gv[:, 0], dtype=jnp.int32)
    gy = jnp.asarray(gv[:, 1], dtype=jnp.int32)
    gz = jnp.asarray(gv[:, 2], dtype=jnp.int32)
    mask = (None if g_mask is None
            else jnp.asarray(g_mask, dtype=psi_r.dtype))
    return fn(psi_r, gx, gy, gz, mask)


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
    """ψ in r-space at the centroid FFT-grid indices ``r_mu``; see docs/architecture/zeta_fit_face_psi_cct.md."""
    ngkmax = int(psi.shape[-1])
    fft_grid_t = tuple(int(s) for s in fft_grid)
    n_rmu = int(np.shape(r_mu)[0])
    kvecs_shape = (None if kvecs_frac is None
                   else tuple(int(s) for s in np.shape(kvecs_frac)))
    out_sharding = _output_sharding(psi, mesh, n_extra_axes=1)
    key = (psi.shape, tuple(g_index.shape), ngkmax, fft_grid_t, n_rmu,
           norm, kvecs_shape, _sharding_key(psi), out_sharding)

    def build():
        ifftn = _local_box_fft(psi, mesh, kind='ifftn', norm=norm)
        if kvecs_frac is None:
            @jax.jit
            def fn(psi_, g_index_, r_mu_):
                rb = ifftn(_box_kernel(psi_, g_index_, ngkmax=ngkmax))
                out = rb[:, :, :, r_mu_[:, 0], r_mu_[:, 1], r_mu_[:, 2]]
                return _maybe_constrain(out, out_sharding)
            return fn
        @jax.jit
        def fn(psi_, g_index_, r_mu_, kvecs_):
            rb = ifftn(_box_kernel(psi_, g_index_, ngkmax=ngkmax))
            rb = apply_bloch_phase(rb, kvecs_, fft_grid_t)
            out = rb[:, :, :, r_mu_[:, 0], r_mu_[:, 1], r_mu_[:, 2]]
            return _maybe_constrain(out, out_sharding)
        return fn

    fn = _cached_jit('to_rmu', key, build)
    g_index_j = _cached_gindex_dev(g_index)
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
    """Per-rank-local body of :func:`to_rchunk`: G-flat → FFT-box → IFFT → r-slice → optional Bloch phase; see docs/architecture/zeta_fit_face_psi_cct.md."""
    ngkmax = int(psi.shape[-1])
    fft_grid_t = tuple(int(s) for s in fft_grid)
    nx, ny, nz = fft_grid_t
    n_rtot = nx * ny * nz
    r_len_i = int(r_len)

    box = _box_kernel(psi, g_index, ngkmax=ngkmax)
    rb = local_ifftn3(box, axes=(-3, -2, -1), norm=norm)
    # Reshape (..., nx, ny, nz) → (..., n_rtot).  Same contract as
    # to_rchunk._local_rchunk: assumes 3 leading axes before the spatial.
    rb_flat = rb.reshape(*rb.shape[:3], n_rtot)
    slab = take_rchunk_padded(rb_flat, r0, r_len_i)
    if kvecs_frac is not None:
        slab = apply_bloch_phase_on_slice(
            slab, kvecs_frac, fft_grid_t, r0, r_len_i)
    return slab


def to_rpoints_inner(
    psi: jax.Array,
    g_index: jax.Array,
    fft_grid: Sequence[int],
    r_flat_idx: jax.Array,
    *,
    norm: str = "backward",
    kvecs_frac: jax.Array | None = None,
) -> jax.Array:
    """The arbitrary-point twin of :func:`to_rchunk_inner`; see docs/architecture/zeta_fit_face_psi_cct.md."""
    ngkmax = int(psi.shape[-1])
    fft_grid_t = tuple(int(s) for s in fft_grid)
    nx, ny, nz = fft_grid_t
    n_rtot = nx * ny * nz

    box = _box_kernel(psi, g_index, ngkmax=ngkmax)
    rb = local_ifftn3(box, axes=(-3, -2, -1), norm=norm)
    # Reshape (..., nx, ny, nz) → (..., n_rtot), then gather the tile's
    # own cells.  Same 3-leading-axes contract as to_rchunk_inner.
    rb_flat = rb.reshape(*rb.shape[:3], n_rtot)
    r_idx = jnp.asarray(r_flat_idx, dtype=jnp.int32)
    slab = jnp.take(rb_flat, jnp.clip(r_idx, 0, n_rtot - 1), axis=-1)
    if kvecs_frac is not None:
        slab = apply_bloch_phase_at(slab, kvecs_frac, fft_grid_t, r_idx)
    return slab


def take_rchunk_padded(values: jax.Array, r0, r_len: int) -> jax.Array:
    """Take a fixed-width flat-r slab, zero-filling beyond physical r; see docs/architecture/zeta_fit_face_psi_cct.md."""
    n_rtot = int(values.shape[-1])
    r_len_i = int(r_len)
    r0_arr = jnp.asarray(r0, dtype=jnp.int32)

    def _direct(_):
        # Preserve the incumbent contiguous-slice executable for every main
        # chunk.  Only the ragged final carrier takes the masked gather arm.
        return jax.lax.dynamic_slice_in_dim(
            values, r0_arr, r_len_i, axis=-1)

    def _padded(_):
        r_idx = r0_arr + jnp.arange(r_len_i, dtype=jnp.int32)
        valid = (r_idx >= 0) & (r_idx < n_rtot)
        safe_idx = jnp.clip(r_idx, 0, n_rtot - 1)
        slab = jnp.take(values, safe_idx, axis=-1)
        valid_shape = (1,) * (slab.ndim - 1) + (r_len_i,)
        return jnp.where(valid.reshape(valid_shape), slab, 0)

    # ``lax.cond`` traces both arms.  A slice wider than the entire operand
    # is statically invalid even when the runtime predicate selects the
    # padded arm, so omit that arm altogether for this tiny-grid case.
    if r_len_i > n_rtot:
        return _padded(None)

    in_bounds = (r0_arr >= 0) & (r0_arr + r_len_i <= n_rtot)
    return jax.lax.cond(in_bounds, _direct, _padded, operand=None)


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
    allow_padded_tail: bool = False,
) -> jax.Array:
    """ψ in r-space on a contiguous flat-r slab ``[r0, r0 + r_len)``; see docs/architecture/zeta_fit_face_psi_cct.md."""
    ngkmax = int(psi.shape[-1])
    fft_grid_t = tuple(int(s) for s in fft_grid)
    nx, ny, nz = fft_grid_t
    n_rtot = nx * ny * nz
    r_len_i = int(r_len)
    if isinstance(r0, (int, np.integer)):
        r0_i = int(r0)
        outside = r0_i < 0 or r0_i + r_len_i > n_rtot
        padded_tail = (bool(allow_padded_tail) and 0 <= r0_i < n_rtot
                       and r0_i + r_len_i > n_rtot)
        if outside and not padded_tail:
            raise ValueError(
                f"to_rchunk: [{r0_i}, {r0_i + r_len_i}) "
                f"out of [0, {n_rtot})"
                + (" (set allow_padded_tail=True only for a "
                   "runtime.padding-derived carrier)"
                   if r0_i + r_len_i > n_rtot else ""))

    psi_spec = P(*_spec_of(psi))
    out_spec = tuple(_output_sharding(psi, mesh, n_extra_axes=1).spec)
    kvecs_shape = (None if kvecs_frac is None
                   else tuple(int(s) for s in np.shape(kvecs_frac)))
    key = (psi.shape, tuple(g_index.shape), ngkmax, fft_grid_t, r_len_i,
           norm, kvecs_shape, _sharding_key(psi), out_spec, id(mesh))

    def build():
        # Per-rank body = ``to_rchunk_inner`` exactly: same _box_kernel +
        # jnp.fft.ifftn + flat-reshape + dynamic-slice (+ phase-on-slice
        # when kvecs_frac is set).  Hoisted as the single source of truth
        # so future changes to the kernel land in one place.
        if kvecs_frac is None:
            @partial(
                shard_map,
                mesh=mesh,
                in_specs=(psi_spec, P(None, None, None, None), P()),
                out_specs=P(*out_spec),
                check_vma=False,
            )
            def _local_rchunk(psi_l, g_index_l, r0_l):
                return to_rchunk_inner(
                    psi_l, g_index_l, fft_grid_t, r0_l, r_len_i, norm=norm)

            @jax.jit
            def fn(psi_, g_index_, r0_):
                return _local_rchunk(psi_, g_index_, r0_)
            return fn

        @partial(
            shard_map,
            mesh=mesh,
            in_specs=(psi_spec, P(None, None, None, None), P(),
                      P(None, None)),
            out_specs=P(*out_spec),
            check_vma=False,
        )
        def _local_rchunk(psi_l, g_index_l, r0_l, kvecs_l):
            return to_rchunk_inner(
                psi_l, g_index_l, fft_grid_t, r0_l, r_len_i,
                norm=norm, kvecs_frac=kvecs_l)

        @jax.jit
        def fn(psi_, g_index_, r0_, kvecs_):
            return _local_rchunk(psi_, g_index_, r0_, kvecs_)
        return fn

    fn = _cached_jit('to_rchunk', key, build)

    g_index_j = _cached_gindex_dev(g_index)
    r0_arg = (jnp.int32(int(r0)) if isinstance(r0, (int, np.integer)) else r0)
    if kvecs_frac is None:
        return fn(psi, g_index_j, r0_arg)
    return fn(psi, g_index_j, r0_arg,
              jnp.asarray(kvecs_frac, dtype=jnp.float64))


def gflat_to_rchunk_aot_memory(
    *,
    mesh: Mesh,
    nk: int,
    band_carrier: int,
    nspinor: int,
    ngkmax: int,
    fft_grid: Sequence[int],
    r_carrier: int,
    norm: str,
    dtype=jnp.complex128,
) -> AotPeakBreakdown:
    """AOT memory breakdown of the canonical full-Bloch WFN r-slab program; see docs/architecture/zeta_fit_face_psi_cct.md."""
    nk = int(nk)
    band_carrier = int(band_carrier)
    nspinor = int(nspinor)
    ngkmax = int(ngkmax)
    r_carrier = int(r_carrier)
    fft_grid_t = tuple(int(v) for v in fft_grid)
    if len(fft_grid_t) != 3 or any(v <= 0 for v in fft_grid_t):
        raise ValueError(
            "gflat_to_rchunk_aot_memory: fft_grid must contain three "
            f"positive extents; got {fft_grid_t}")
    if min(nk, band_carrier, nspinor, ngkmax, r_carrier) <= 0:
        raise ValueError(
            "gflat_to_rchunk_aot_memory: all logical extents must be "
            "positive")
    if norm not in ("backward", "ortho", "forward"):
        raise ValueError(
            "gflat_to_rchunk_aot_memory: norm must be 'backward', "
            f"'ortho', or 'forward'; got {norm!r}")

    platform = mesh.devices.flat[0].platform
    key = (
        id(mesh), nk, band_carrier, nspinor, ngkmax, fft_grid_t,
        r_carrier, norm, jnp.dtype(dtype).str, platform,
        tuple(mesh.axis_names),
        tuple(int(mesh.shape[a]) for a in mesh.axis_names),
    )
    hit = _RCHUNK_PEAK_CACHE.get(key)
    if hit is not None:
        return hit

    psi_sharding = NamedSharding(mesh, band_sphere_spec())
    gindex_sharding = NamedSharding(mesh, P(None, None, None, None))
    kvec_sharding = NamedSharding(mesh, P(None, None))
    rep = NamedSharding(mesh, P())

    @partial(
        shard_map,
        mesh=mesh,
        in_specs=(band_sphere_spec(), P(None, None, None, None), P(),
                  P(None, None)),
        out_specs=band_sphere_spec(),
        check_vma=False,
    )
    def _local(psi_G, g_index, r0, kvecs_frac):
        return to_rchunk_inner(
            psi_G, g_index, fft_grid_t, r0, r_carrier,
            norm=norm, kvecs_frac=kvecs_frac)

    kernel = jax.jit(
        _local,
        in_shardings=(psi_sharding, gindex_sharding, rep, kvec_sharding),
        out_shardings=psi_sharding,
    )
    specs = (
        jax.ShapeDtypeStruct(
            (nk, band_carrier, nspinor, ngkmax), dtype,
            sharding=psi_sharding),
        jax.ShapeDtypeStruct(
            (nk, *fft_grid_t), jnp.int32, sharding=gindex_sharding),
        jax.ShapeDtypeStruct((), jnp.int32, sharding=rep),
        jax.ShapeDtypeStruct((nk, 3), jnp.float64, sharding=kvec_sharding),
    )
    lowered = kernel.lower(*specs)
    compiled = lowered.compile(
        compiler_options={"xla_gpu_memory_limit_slop_factor": 10000}
    ) if platform in ("gpu", "cuda") else lowered.compile()
    from runtime.aot_memory import aot_kernel_peak_bytes
    memory = aot_kernel_peak_bytes(compiled, platform=platform)
    if platform in ("gpu", "cuda") and not memory.fft_specs:
        raise RuntimeError(
            "gflat_to_rchunk_aot_memory: the compiled canonical WFN "
            "r-slab program exposes no FFT operation, so its cuFFT workspace "
            "cannot be certified")
    if platform in ("gpu", "cuda") and not memory.cufft_measured:
        raise RuntimeError(
            "gflat_to_rchunk_aot_memory: the canonical WFN r-slab program's "
            "cuFFT workspace query is unavailable on CUDA; refusing a "
            "known-low memory preflight")
    _RCHUNK_PEAK_CACHE[key] = memory
    return memory


def gflat_to_rchunk_aot_peak_bytes(
    *,
    mesh: Mesh,
    nk: int,
    band_carrier: int,
    nspinor: int,
    ngkmax: int,
    fft_grid: Sequence[int],
    r_carrier: int,
    norm: str,
    dtype=jnp.complex128,
) -> int:
    """Per-rank total peak HBM for the canonical WFN r-slab program; see docs/architecture/zeta_fit_face_psi_cct.md."""
    return int(gflat_to_rchunk_aot_memory(
        mesh=mesh, nk=nk, band_carrier=band_carrier, nspinor=nspinor,
        ngkmax=ngkmax, fft_grid=fft_grid, r_carrier=r_carrier, norm=norm,
        dtype=dtype).total)



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
    """Per-rank-local body of :func:`to_rmu`: G-flat → FFT-box → IFFT → centroid sample → optional Bloch phase; see docs/architecture/zeta_fit_face_psi_cct.md."""
    ngkmax = int(psi.shape[-1])
    fft_grid_t = tuple(int(s) for s in fft_grid)
    box = _box_kernel(psi, g_index, ngkmax=ngkmax)
    rb = local_ifftn3(box, axes=(-3, -2, -1), norm=norm)
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


def gflat_to_rmu(
    psi_G: jax.Array,
    g_index: np.ndarray | jax.Array,
    r_mu: np.ndarray | jax.Array,
    *,
    mesh: Mesh,
    fft_grid: Sequence[int],
    kvecs_frac: np.ndarray | jax.Array | None = None,
    k_row_map: np.ndarray | jax.Array | None = None,
    norm: str = "backward",
    chunk_size: int | None = None,
) -> jax.Array:
    """ψ(G-flat) → ψ at centroid grid points, fused over all (k, n); see docs/architecture/zeta_fit_face_psi_cct.md."""
    fft_grid_t = tuple(int(s) for s in fft_grid)
    nx, ny, nz = fft_grid_t
    n_rtot = nx * ny * nz
    nk        = int(psi_G.shape[0])
    nb_total  = int(psi_G.shape[1])
    ns        = int(psi_G.shape[2])
    ngkmax    = int(psi_G.shape[3])
    r_mu_shape = tuple(int(s) for s in np.shape(r_mu))
    if len(r_mu_shape) != 2 or r_mu_shape[1] != 3:
        raise ValueError(
            f"gflat_to_rmu: r_mu must be (n_rmu, 3); got shape "
            f"{r_mu_shape}.")
    n_rmu     = r_mu_shape[0]
    p_prod    = spec_divisor(mesh, band_sphere_spec(), axis=1)
    # Band-flat sharding needs the band axis divisible by the mesh.  Pad it
    # up with ZERO bands so ANY device count works: the htransform SP /
    # galerkin entry (bandstructure.htransform.streaming_galerkin_solve)
    # passes an un-rounded band window (e.g. nb=40 on a 16-device mesh),
    # unlike the GW path which pre-rounds via Meta._round_up(world_size).
    # Pad bands are ψ=0 ⇒ zero centroid samples, dropped from the output
    # below.  No-op when nb_total already divides p_prod (nb_pad_total ==
    # nb_total): single-node / divisible meshes stay byte-identical.
    # ``runtime.padding.pad_axis`` is THE band pad, shared with
    # ``common.mtxel_sweep`` — same arithmetic, same zero-band argument, one
    # implementation.  This call site used to inline the round_up + jnp.pad.
    psi_G = pad_axis(psi_G, p_prod, axis=1).array
    nb_pad_total = int(psi_G.shape[1])
    nb_local = nb_pad_total // p_prod
    N        = nk * nb_local

    # Round-6 canonical-accessor path: accept either a numpy g_index
    # OR the loader's cached ``WfnLoader.box_index_dev(...)`` jax.Array
    # without a device→host roundtrip.  Validation uses ``.shape`` /
    # ``.ndim`` (both supported by numpy and jax.Array natively).
    g_shape = tuple(int(s) for s in np.shape(g_index))
    if len(g_shape) != 4 or g_shape[0] != nk:
        raise ValueError(
            f"gflat_to_rmu: g_index must be (nk, nx, ny, nz); got "
            f"shape {g_shape}, expected nk={nk}.")
    if g_shape[1:] != fft_grid_t:
        raise ValueError(
            f"gflat_to_rmu: g_index trailing shape {g_shape[1:]} "
            f"≠ fft_grid {fft_grid_t}.")

    # Retain the eager caller guard without forcing a device→host roundtrip
    # when the canonical producer already supplies a jax.Array.  Device
    # values remain runtime operands below, exactly as in ``to_rmu``.
    if not isinstance(r_mu, jax.Array):
        r_mu_host = np.asarray(r_mu, dtype=np.int32)
        if (np.any(r_mu_host[:, 0] < 0) or np.any(r_mu_host[:, 0] >= nx)
                or np.any(r_mu_host[:, 1] < 0)
                or np.any(r_mu_host[:, 1] >= ny)
                or np.any(r_mu_host[:, 2] < 0)
                or np.any(r_mu_host[:, 2] >= nz)):
            raise ValueError(
                f"gflat_to_rmu: r_mu has out-of-range coords for "
                f"fft_grid {fft_grid_t}.")

    # Clamp the chunk to the actual row count: a chunk larger than the
    # data only inflates the flat-axis zero-pad — and with it the
    # per-iteration FFT box, which is sized cs·ns·n_rtot·16 B REGARDLESS
    # of N (measured: the 3×3 nb=80 refit galerkin has N = 720 rows but
    # an HBM-budget cs of 6103 → an 8.5× padded box, a 16.76 GiB fused
    # alloc for ~1 GB of data).  No-op whenever cs ≤ N (production
    # scale); pad rows are zeros truncated at out[:N] either way.
    cs       = max(1, min(int(chunk_size if chunk_size else N), N))
    n_chunks = (N + cs - 1) // cs
    pad_N    = n_chunks * cs - N

    if kvecs_frac is None:
        kvecs_shape = None
        kvecs_dev = None
    else:
        # Shape validation is host-only, but the values remain on device.
        # A streamed tile is already a jax.Array; np.asarray here would add a
        # device-to-host-to-device copy at every k tile boundary.
        kvecs_shape = tuple(int(s) for s in np.shape(kvecs_frac))
        if kvecs_shape != (nk, 3):
            raise ValueError(
                f"gflat_to_rmu: kvecs_frac must be ({nk}, 3); got "
                f"{kvecs_shape}.")
        kvecs_dev = jnp.asarray(kvecs_frac, dtype=jnp.float64)
    if k_row_map is None:
        k_row_map_shape = None
        k_row_map_dev = None
    else:
        k_row_map_shape = tuple(int(s) for s in np.shape(k_row_map))
        if k_row_map_shape != (nk,):
            raise ValueError(
                f"gflat_to_rmu: k_row_map must be ({nk},); got "
                f"{k_row_map_shape}.")
        k_row_map_host = np.asarray(k_row_map, dtype=np.int64)
        if np.any(k_row_map_host < 0) or np.any(k_row_map_host >= nk):
            raise ValueError(
                f"gflat_to_rmu: k_row_map contains a row outside [0,{nk}).")
        k_row_map_dev = jnp.asarray(k_row_map, dtype=jnp.int32)
    # Resolve the canonical device buffer without a numpy roundtrip for
    # jax.Array inputs.  See _resolve_gindex_dev.
    g_index_dev_canonical, _ = _resolve_gindex_dev(g_index)
    r_mu_dev = jnp.asarray(r_mu, dtype=jnp.int32)

    key = (
        tuple(int(s) for s in psi_G.shape), tuple(g_shape),
        fft_grid_t, r_mu_shape, ngkmax,
        norm, kvecs_shape, k_row_map_shape, cs, n_chunks, pad_N,
        # The shard_map below owns the input layout: every operand enters as
        # ``band_sphere_spec()`` on this explicit mesh.  Key that contract,
        # not the incidental sharding wrapper returned by each streamed
        # loader call; recent JAX can represent equal placements with
        # distinct wrappers/spec spellings and would otherwise grow one
        # compiled family per tile.
        (id(mesh), tuple(band_sphere_spec())),
    )

    def build():
        # G-index, centroid coordinates and k-vectors are runtime operands:
        # fixed-shape streamed tiles and centroid sets share one executable,
        # with no retained per-content device constant.

        # Round-6: pass g_index through shard_map's in_specs (NOT
        # closure capture) so the Auto-sharded NamedSharding-replicated
        # canonical buffer from ``loader.box_index_dev(...)`` is
        # Manual-mode-compatible inside the kernel.  Pre-Round-6 the
        # closure captured a SingleDeviceSharding jax.Array (no-mesh)
        # produced by ``_cached_gindex_dev``'s ``jnp.asarray`` — which
        # left the wfn_transforms-side buffer DIVORCED from the
        # WfnLoader-side canonical buffer (different sharding ⇒
        # different device allocation).  Threading it as a shard_map
        # input with ``P(None,None,None,None)`` matches the pattern
        # already used by ``isdf_fitting._make_pair_pipeline_sm`` for
        # the same buffer and ensures both sides share the canonical
        # WfnLoader-cached device allocation.
        in_spec  = band_sphere_spec()
        gidx_spec = P(None, None, None, None)
        rmu_spec = P(None, None)
        kvec_spec = P(None, None)
        out_spec = band_sphere_spec()

        def _body(psi_, g_index_, r_mu_, kvecs_, k_row_map_):
            # Per-rank: (nk, nb_local, ns, ngkmax).
            psi_flat = psi_.reshape(N, ns, ngkmax)
            if pad_N:
                psi_flat = jnp.pad(
                    psi_flat, ((0, pad_N), (0, 0), (0, 0)))
            out_flat = jnp.zeros(
                (N + pad_N, ns, n_rmu), dtype=psi_.dtype)
            if kvecs_ is not None:
                # Preserve the established separable phase arithmetic exactly;
                # only its k-vector operand moved from a closure constant to a
                # runtime value so fixed-shape k tiles share one executable.
                _ph = lambda k_axis, n: jnp.exp(
                    +2j * jnp.pi * k_axis[:, None]
                    * (jnp.arange(n) / n)[None, :])
                phx = _ph(kvecs_[:, 0], nx)
                phy = _ph(kvecs_[:, 1], ny)
                phz = _ph(kvecs_[:, 2], nz)
                phx_rmu_all = phx[:, r_mu_[:, 0]]
                phy_rmu_all = phy[:, r_mu_[:, 1]]
                phz_rmu_all = phz[:, r_mu_[:, 2]]
            else:
                phx_rmu_all = phy_rmu_all = phz_rmu_all = None

            def body(out, i):
                i0    = i * cs
                sub   = jax.lax.dynamic_slice_in_dim(
                    psi_flat, i0, cs, axis=0)            # (cs, ns, ngkmax)
                k_row = jnp.clip(
                    (i0 + jnp.arange(cs)) // nb_local, 0, nk - 1)  # (cs,)
                mapped_k_row = (k_row if k_row_map_ is None
                                else k_row_map_[k_row])
                # Singleton-nb reshape so _box_kernel's (n_k, nb, ns,
                # ngkmax) contract takes (cs, 1, ns, ngkmax) per row.
                # Per-row g_index gather: (cs, nx, ny, nz).
                sub4 = sub.reshape(cs, 1, ns, ngkmax)
                g_per_row = g_index_[mapped_k_row]
                box = _box_kernel(
                    sub4, g_per_row, ngkmax=ngkmax)      # (cs, 1, ns, nx, ny, nz)
                box = box.reshape(cs, ns, nx, ny, nz)
                rb = local_ifftn3(box, axes=(-3, -2, -1), norm=norm)
                # Centroid gather — (cs, ns, n_rmu).
                samples = rb[:, :, r_mu_[:, 0], r_mu_[:, 1], r_mu_[:, 2]]
                if phx_rmu_all is not None:
                    # Per-row Bloch phase at the gathered centroid
                    # cells — apply_bloch_phase is multiplicative on the
                    # spatial axes, so applying it post-gather is
                    # algebraically identical to applying it on the
                    # full FFT box before gather (cf. gflat_to_rchunk's
                    # phase-on-slice pattern).
                    phx_q = phx_rmu_all[mapped_k_row]
                    phy_q = phy_rmu_all[mapped_k_row]
                    phz_q = phz_rmu_all[mapped_k_row]
                    samples = samples * (phx_q * phy_q * phz_q)[:, None, :]
                return jax.lax.dynamic_update_slice_in_dim(
                    out, samples, i0, axis=0), None

            out_flat, _ = jax.lax.scan(
                body, out_flat, jnp.arange(n_chunks, dtype=jnp.int32), unroll=1)
            if pad_N:
                out_flat = out_flat[:N]
            return out_flat.reshape(nk, nb_local, ns, n_rmu)

        if kvecs_frac is None and k_row_map is None:
            @partial(shard_map, mesh=mesh,
                     in_specs=(in_spec, gidx_spec, rmu_spec),
                     out_specs=out_spec,
                     check_vma=False)
            def _kernel(psi_, g_index_, r_mu_):
                return _body(psi_, g_index_, r_mu_, None, None)
        elif k_row_map is None:
            @partial(shard_map, mesh=mesh,
                     in_specs=(in_spec, gidx_spec, rmu_spec, kvec_spec),
                     out_specs=out_spec,
                     check_vma=False)
            def _kernel(psi_, g_index_, r_mu_, kvecs_):
                return _body(psi_, g_index_, r_mu_, kvecs_, None)
        elif kvecs_frac is None:
            @partial(shard_map, mesh=mesh,
                     in_specs=(in_spec, gidx_spec, rmu_spec, P(None)),
                     out_specs=out_spec,
                     check_vma=False)
            def _kernel(psi_, g_index_, r_mu_, k_row_map_):
                return _body(psi_, g_index_, r_mu_, None, k_row_map_)
        else:
            @partial(shard_map, mesh=mesh,
                     in_specs=(in_spec, gidx_spec, rmu_spec, kvec_spec,
                               P(None)),
                     out_specs=out_spec,
                     check_vma=False)
            def _kernel(psi_, g_index_, r_mu_, kvecs_, k_row_map_):
                return _body(
                    psi_, g_index_, r_mu_, kvecs_, k_row_map_)

        return jax.jit(_kernel)

    fn = _cached_jit('gflat_to_rmu', key, build)
    if kvecs_frac is None and k_row_map is None:
        out = fn(psi_G, g_index_dev_canonical, r_mu_dev)
    elif k_row_map is None:
        out = fn(psi_G, g_index_dev_canonical, r_mu_dev, kvecs_dev)
    elif kvecs_frac is None:
        out = fn(psi_G, g_index_dev_canonical, r_mu_dev, k_row_map_dev)
    else:
        out = fn(
            psi_G, g_index_dev_canonical, r_mu_dev, kvecs_dev,
            k_row_map_dev)
    if nb_pad_total != nb_total:
        # Drop the zero pad-bands.  Replicate the band axis first (bands off
        # the mesh) so the slice to the logical nb_total — which need not
        # divide the mesh — is well defined; the sole caller
        # (load_centroids_band_chunked) reshards immediately afterwards, so
        # this transient gather is not a lasting replicated buffer.
        out = jax.lax.with_sharding_constraint(out, NamedSharding(mesh, P()))
        out = out[:, :nb_total]
    return out


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

def accumulate_rchunk_to_gflat(
    rchunk: jax.Array,
    gflat_acc: jax.Array,
    *,
    mesh: Mesh,
    fft_grid: Sequence[int],
    r0=None,
    sphere_idx: np.ndarray | jax.Array,
    qvec_frac: np.ndarray | jax.Array | None = None,
    norm: str = "backward",
    chunk_size: int | None = None,
    r_indices: jax.Array | None = None,
) -> jax.Array:
    """Add ``FFT(pad(phase(rchunk)))[sphere_idx]`` into ``gflat_acc``; see docs/architecture/zeta_fit_face_psi_cct.md."""
    indexed = r_indices is not None
    if indexed == (r0 is not None):
        raise ValueError(
            "accumulate_rchunk_to_gflat: pass exactly one of r0 / r_indices "
            f"(got r0={r0!r}, r_indices="
            f"{'an array' if indexed else None}).")

    fft_grid_t = tuple(int(s) for s in fft_grid)
    nx, ny, nz = fft_grid_t
    n_rtot = nx * ny * nz
    n_q       = int(rchunk.shape[0])
    n_rmu_pad = int(rchunk.shape[1])
    r_len_i   = int(rchunk.shape[-1])
    mu_gflat_spec = P(None, ('x', 'y'), None)
    p_prod    = spec_divisor(mesh, mu_gflat_spec, axis=1)
    from runtime.padding import authenticate_padded_axis
    authenticate_padded_axis(
        n_rmu_pad, n_rmu_pad, mesh,
        name="gflat centroid carrier", spec=mu_gflat_spec, axis=1)
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

    # Clamp the chunk to the actual row count: a chunk larger than the
    # data only inflates the flat-axis zero-pad — and with it the
    # per-iteration FFT box, which is sized cs·ns·n_rtot·16 B REGARDLESS
    # of N (measured: the 3×3 nb=80 refit galerkin has N = 720 rows but
    # an HBM-budget cs of 6103 → an 8.5× padded box, a 16.76 GiB fused
    # alloc for ~1 GB of data).  No-op whenever cs ≤ N (production
    # scale); pad rows are zeros truncated at out[:N] either way.
    cs       = max(1, min(int(chunk_size if chunk_size else N), N))
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
        norm, qvec_shape, qvec_id, cs, n_chunks, pad_N, indexed,
        _sharding_key(rchunk), _sharding_key(gflat_acc),
    )

    def build():
        # Per-q tables baked into the closure as constants.
        # Share the device-resident sphere idx across cache_key
        # variants (different qvec/r_chunk metadata can build distinct
        # closures, but the sphere idx itself is content-stable when
        # the WFN is fixed).  Same dedup principle as for g_index above.
        sphere_c = _cached_gindex_dev(sphere_arr)
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
        in_spec = out_spec = mu_gflat_spec

        @partial(shard_map, mesh=mesh,
                 in_specs=(in_spec, in_spec, P()),
                 out_specs=out_spec,
                 check_vma=False)
        def _kernel(rch_, acc_, r_):
            # Per-rank: (n_q, n_mu_local, r_len) / (n_q, n_mu_local, ngkmax).
            # ``r_`` is the flat-r start scalar (r0 path) or the (r_len,)
            # index table (r_indices path); both arrive replicated.
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
                r_idx_slab = (jnp.clip(r_, 0, n_rtot - 1) if indexed
                              else r_ + jnp.arange(r_len_i, dtype=jnp.int32))
                ny_nz = jnp.int32(ny * nz)
                rx_slab = r_idx_slab // ny_nz
                ry_slab = (r_idx_slab // jnp.int32(nz)) % jnp.int32(ny)
                rz_slab = r_idx_slab %  jnp.int32(nz)
            if indexed:
                # The slab-to-box placement as a GATHER, not a scatter: the
                # inverse table ``box_from_slab[r]`` names the slab column
                # holding grid cell ``r`` (or the appended zero column).  It
                # is built once per call from the (r_len,) index table —
                # every real index is distinct, and pad sentinels lie outside
                # the box so the drop leaves their cells pointing at the
                # zero column.  A scatter into the (cs, n_rtot) box was
                # measured at 1.27 s per tile against 5 ms for the slab
                # path (Si P4, 2026-09-05); this gather restores that class.
                box_from_slab = jnp.full((n_rtot,), r_len_i, dtype=jnp.int32)
                box_from_slab = box_from_slab.at[r_].set(
                    jnp.arange(r_len_i, dtype=jnp.int32), mode='drop',
                    unique_indices=True)

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
                if indexed:
                    sub_pad = jnp.concatenate(
                        [sub, jnp.zeros((cs, 1), dtype=sub.dtype)], axis=-1)
                    # Every table entry lies in [0, r_len]: real cells name
                    # their slab column, pads name the zero column.
                    buf = jnp.take(sub_pad, box_from_slab, axis=-1)
                else:
                    buf = jnp.zeros((cs, n_rtot), dtype=sub.dtype)
                    buf = jax.lax.dynamic_update_slice_in_dim(
                        buf, sub, r_, axis=-1)
                box = buf.reshape(cs, nx, ny, nz)
                G = local_fftn3(box, axes=(-3, -2, -1), norm=norm).reshape(cs, n_rtot)
                contrib = jnp.take_along_axis(
                    G, sphere_c[q_row], axis=-1, mode='promise_in_bounds')
                acc_sub = jax.lax.dynamic_slice_in_dim(acc, i0, cs, axis=0)
                return jax.lax.dynamic_update_slice_in_dim(
                    acc, acc_sub + contrib, i0, axis=0), None

            acc_flat, _ = jax.lax.scan(
                body, acc_flat, jnp.arange(n_chunks, dtype=jnp.int32), unroll=1)
            if pad_N:
                acc_flat = acc_flat[:N]
            return acc_flat.reshape(n_q, n_mu_local, ngkmax)

        return jax.jit(_kernel, donate_argnums=(1,))

    fn = _cached_jit('accumulate_rchunk_to_gflat', key, build)
    if indexed:
        r_arg = jnp.asarray(r_indices, dtype=jnp.int32)
    else:
        r_arg = (jnp.int32(int(r0)) if isinstance(r0, (int, np.integer)) else r0)
    return fn(rchunk, gflat_acc, r_arg)


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
#   * ζ-r → G FFT (pre-FFT, sign='-'):           gw.isdf_fitting, the ζ WRITER
#     for ``z_q,μ(r) = exp(-2πi q·r) ζ_q,μ(r)``
#     before scattering onto the (q + G) sphere.  It was also the READER's
#     ``zeta_loader._do_disk_to_G`` until 2026-08-07, when that path was
#     deleted: the writer emits G-flat only, so nothing reads r-space ζ.

def apply_bloch_phase(
    box: jax.Array,
    kvecs_frac: jax.Array,
    fft_grid: tuple[int, int, int],
    *,
    sign: int = 1,
) -> jax.Array:
    """box × exp(sign · 2πi k·r) applied as three separable 1D multiplies; see docs/architecture/zeta_fit_face_psi_cct.md."""
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


def apply_bloch_phase_at(
    slab: jax.Array,
    kvecs_frac: jax.Array,
    fft_grid: tuple[int, int, int],
    flat_idx: jax.Array,
    *,
    sign: int = 1,
) -> jax.Array:
    """``slab × exp(sign·2πi k·r)`` at arbitrary flat-r grid points; see docs/architecture/zeta_fit_face_psi_cct.md."""
    nx, ny, nz = (int(s) for s in fft_grid)
    r_len_i = int(flat_idx.shape[-1])

    fx = jnp.arange(nx, dtype=jnp.float64) / nx
    fy = jnp.arange(ny, dtype=jnp.float64) / ny
    fz = jnp.arange(nz, dtype=jnp.float64) / nz
    scale = jnp.complex128(int(sign) * 2j * jnp.pi)
    px = jnp.exp(scale * kvecs_frac[:, 0:1] * fx[None, :])    # (n_k, nx)
    py = jnp.exp(scale * kvecs_frac[:, 1:2] * fy[None, :])
    pz = jnp.exp(scale * kvecs_frac[:, 2:3] * fz[None, :])

    # Decode r_flat → (rx, ry, rz) on the slab.  ``ny``/``nz`` are static
    # so the divmod constants fold cleanly whether ``flat_idx`` is a
    # constant table or a traced one.
    flat = jnp.clip(flat_idx, 0, nx * ny * nz - 1)           # (r_len,)
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


def apply_bloch_phase_on_slice(
    slab: jax.Array,
    kvecs_frac: jax.Array,
    fft_grid: tuple[int, int, int],
    r0,
    r_len: int,
    *,
    sign: int = 1,
) -> jax.Array:
    """``slab × exp(sign·2πi k·r)`` over a contiguous flat-r slab; see docs/architecture/zeta_fit_face_psi_cct.md."""
    r_len_i = int(r_len)
    return apply_bloch_phase_at(
        slab, kvecs_frac, fft_grid,
        r0 + jnp.arange(r_len_i, dtype=jnp.int32), sign=sign)


# ===========================================================================
# ψ(G)-loading helpers (relocated from the former common/load_wfns.py).
#
# Each takes a :class:`wfn_loader.WfnLoader` as ``wfn`` and composes
# ``wfn.load(...)`` with the transforms above.  The legacy ``sym`` argument
# is unused (WfnLoader builds its own SymMaps) but kept for caller-API
# back-compat.  Bodies are byte-for-byte the load_wfns originals; only the
# module-self imports were dropped (to_box / to_rchunk / gflat_to_rmu now
# live here) and ``timing`` is imported locally.
# ===========================================================================


# THE 1×1 ('x','y') mesh over this process's own device — an ALIAS, not a
# definition.  This module used to own it, and ``common.collectives`` (the
# cross-process service) imported it from here, which is a device-layer
# module reaching up into a ψ-transform kernel for its own topology.  The
# owner moved on 2026-07-31; the function object is unchanged, so the
# identity contract that makes this work is unchanged too.
#
# THAT IDENTITY IS LOAD-BEARING.  The shape-keyed jit caches in this module
# (``to_box`` and friends) key on the output ``NamedSharding``, which embeds
# the mesh OBJECT.  Two equal-but-distinct 1×1 ``Mesh``es double every one of
# them.  So this MUST stay an alias — re-implementing it here, however
# faithfully, is the bug.  ``tests/test_layering.py`` pins the direction and
# ``tests/test_collectives_distribution.py`` pins the identity.
#
# It is the mesh half of the ``WfnLoader.load_process_local`` contract; see
# ``common.collectives.single_device_mesh`` for the ``jax.devices()[:1]``
# hazard it replaces.
from common.collectives import single_device_mesh as process_local_mesh


def _refuse_spinor_zero_fill(ns_want: int, ns_have: int, *, origin: str):
    """Refuse a ψ spinor-extent mismatch instead of zero-filling it; see docs/architecture/zeta_fit_face_psi_cct.md."""
    raise ValueError(
        f"{origin}: refusing to zero-fill the spinor axis from {ns_have} to "
        f"{ns_want} components. meta.nspinor={ns_want} means this is a "
        f"bispinor deck, but the loader was asked for {ns_have} components, "
        f"so the small components psi_S = (alpha/2)(sigma.(k+G)) psi_L would "
        f"be silently replaced by zeros (~4e-4 relative in rho and V_H, below "
        f"every tolerance that would catch it). Pass bispinor=True to this "
        f"loader call so the small components are LIFTED, not invented. See "
        f"common.bispinor_init.lift_to_4spinor.")


def load_kpoint_fftbox_local(wfn, meta, k_idx, nb, *, b_lo: int = 0,
                             bispinor: bool = False,
                             bispinor_lift: str = "raw"):
    """One k-point's ψ in the FFT box, **process-local**; see docs/architecture/zeta_fit_face_psi_cct.md."""
    loader = wfn  # reuse top-level WfnLoader; do NOT re-open (would re-slurp coeffs)
    psi = loader.load_process_local(
        bands=(int(b_lo), int(nb)), k=[int(k_idx)], bispinor=bool(bispinor),
        bispinor_lift=bispinor_lift)
    ns_have = int(psi.shape[2])
    if int(meta.nspinor) > ns_have:
        _refuse_spinor_zero_fill(int(meta.nspinor), ns_have,
                                 origin="wfn_transforms.load_kpoint_fftbox_local")
    psi_box = to_box(psi, loader.box_index(k=[int(k_idx)]),
                     tuple(int(s) for s in meta.fft_grid),
                     mesh=process_local_mesh())
    return psi_box[0]                                # strip the singleton k-axis


def load_kpoint_fftbox(wfn, sym, meta, k_idx, nb):
    """Load a single k-point's wavefunction into the FFT box on GPU; see docs/architecture/zeta_fit_face_psi_cct.md."""
    del sym
    return load_kpoint_fftbox_local(wfn, meta, k_idx, nb)


def get_enk_bandrange(wfn, sym, bandrange, sigma_bandrange, nspinor=None):
    """Return band energies and per-band weights for a given band window; see docs/architecture/zeta_fit_face_psi_cct.md."""
    nspinor = int(wfn.nspinor) if nspinor is None else int(nspinor)
    # Energies are stored on irreducible k; expand to full k using mapping.
    band_lo = int(bandrange[0])
    band_hi = int(bandrange[1])
    nb = band_hi - band_lo
    irk_to_k = np.asarray(wfn.symmetry().irr_idx_k)
    # Handle file-short case (band_hi > nbnd in WFN.h5): read what's
    # available, sentinel-fill the rest so f_n=step(E_F-e)=0 for padded
    # bands.  Using a finite "max(real e) + 1 Ry" instead of ∞ keeps
    # PPM resolvent arithmetic 1/(ω - e + iη) safe under fp warnings.
    nb_in_file = int(wfn.energies.shape[2])
    band_hi_eff = min(band_hi, nb_in_file)
    en_irk = np.asarray(
        wfn.energies[0, :, band_lo:band_hi_eff], dtype=np.float64)
    if band_hi_eff < band_hi:
        sentinel = float(np.asarray(wfn.energies[0, :, :]).max()) + 1.0
        pad = np.full((en_irk.shape[0], band_hi - band_hi_eff), sentinel,
                      dtype=np.float64)
        en_irk = np.concatenate([en_irk, pad], axis=1)
    enk = en_irk[irk_to_k, :]                                   # (nk_full, nb)

    # Weighting heuristic: 1/sqrt(Ec - E) for conduction, 1/sqrt(E - Ev) for
    # valence, capped and normalized; sigma band window set to exactly 1.
    sigma_lo = int(sigma_bandrange[0])
    sigma_hi = int(sigma_bandrange[1])
    enk_sigma_lo = max(sigma_lo - band_lo, 0)
    enk_sigma_hi = min(sigma_hi - band_lo, nb)
    energies_full = np.asarray(wfn.energies[0, :, :], dtype=np.float64)[irk_to_k, :]
    energies_sigma = energies_full[:, sigma_lo:sigma_hi]
    E_min = float(energies_sigma.min())
    E_max = float(energies_sigma.max())
    efermi = float(wfn.efermi)

    val_weights = 1.0 / np.sqrt(np.maximum(E_max - enk, 1e-12))
    cond_weights = 1.0 / np.sqrt(np.maximum(enk - E_min, 1e-12))
    weights = np.where(enk <= efermi, val_weights, cond_weights)
    wmax = weights.max()
    if wmax > 0:
        weights = weights / wmax
    weights[:, enk_sigma_lo:enk_sigma_hi] = 1.0
    weights = np.repeat(weights, repeats=nspinor, axis=1)

    return jnp.asarray(enk), jnp.asarray(weights)


def read_Gvecs_to_devices(
    wfn, sym, bandrange, meta: "Meta", bispinor: bool, mesh_xy: Mesh,
    k_range: tuple[int, int] | None = None,
    *, bispinor_lift: str = "raw",
):
    """G-space wfns on a 2-D mesh, band-sharded, scattered to FFT box; see docs/architecture/zeta_fit_face_psi_cct.md."""
    del sym

    b_lo, b_hi = int(bandrange[0]), int(bandrange[1])
    nb_logical = b_hi - b_lo
    if k_range is None:
        k = "full_bz"
    else:
        k = list(range(int(k_range[0]), int(k_range[1])))

    sharding = band_sphere_spec()

    loader = wfn  # reuse top-level WfnLoader
    psi_G_flat = loader.load(
        bands=(b_lo, b_hi), k=k, sharding=sharding,
        bispinor=bool(bispinor), bispinor_lift=bispinor_lift,
    )
    ns_after_lift = 4 if bispinor else int(loader.nspinor)
    if int(meta.nspinor) > ns_after_lift:
        _refuse_spinor_zero_fill(int(meta.nspinor), ns_after_lift,
                                 origin="wfn_transforms.read_Gvecs_to_devices")
    psi_box = to_box(psi_G_flat, loader.box_index(k=k),
                      tuple(int(s) for s in meta.fft_grid),
                      mesh=mesh_xy)
    return psi_box, nb_logical


def load_psi_gflat_padded(
    loader,
    bands: tuple[int, int],
    *,
    mesh_xy: Mesh,
    bispinor: bool,
    pad_to: int | None = None,
    k="full_bz",
    sharding: P = band_sphere_spec(),
    bispinor_lift: str = "raw",
) -> "jax.Array | None":
    """One capped + zero-padded ψ(G-flat) load — THE shared load dance; see docs/architecture/zeta_fit_face_psi_cct.md."""
    b_lo, b_hi = int(bands[0]), int(bands[1])
    nb_total = b_hi - b_lo
    target = max(nb_total, int(pad_to)) if pad_to is not None else nb_total
    file_nbands = int(loader.nbands)
    b_hi_in_file = min(b_hi, file_nbands)
    if b_lo >= b_hi_in_file:
        return None
    psi = loader.load(
        bands=(b_lo, b_hi_in_file), k=k, sharding=sharding,
        bispinor=bool(bispinor), bispinor_lift=bispinor_lift)
    nb_loaded = int(psi.shape[1])
    if nb_loaded < target:
        psi = jnp.concatenate(
            [psi,
             jnp.zeros((psi.shape[0], target - nb_loaded,
                        psi.shape[2], psi.shape[3]), dtype=psi.dtype)],
            axis=1)
        psi = jax.lax.with_sharding_constraint(
            psi, NamedSharding(mesh_xy, sharding))
    return psi


# ============================================================================
# R-CHUNK EXTRACTION: Contiguous r-space chunking via flattened r-index
# ============================================================================
# R-chunking advantage: r in [r_start, r_end) is contiguous in r-space and can
# be written to HDF5 in a single sequential operation. This allows arbitrary
# chunk sizes by slicing along the flattened xyz index.
# ============================================================================


def prepare_rchunk_carrier(
    mesh_xy: Mesh,
    *,
    r_start: int,
    r_end: int,
    n_rtot: int,
    product_r_spec: P | None,
):
    """Plan and finish the canonical band-sharded r-chunk carrier; see docs/architecture/zeta_fit_face_psi_cct.md."""
    r_start = int(r_start)
    r_end = int(r_end)
    n_rtot = int(n_rtot)
    if not 0 <= r_start < r_end <= n_rtot:
        raise ValueError(
            "prepare_rchunk_carrier: expected "
            f"0 <= r_start < r_end <= {n_rtot}, got "
            f"[{r_start}, {r_end})")

    out_y = NamedSharding(mesh_xy, P(None, None, None, 'y'))
    product_reshard = None
    if product_r_spec is not None:
        expected = P(None, None, None, ('y', 'x'))
        if product_r_spec != expected:
            raise ValueError(
                "prepare_rchunk_carrier: the canonical product-r route "
                f"has spec {expected}, got {product_r_spec}")
        r_pad_divisor = spec_divisor(mesh_xy, product_r_spec, axis=3)
        product_reshard = band_to_product_r_reshard(mesh_xy)
        out_r = NamedSharding(mesh_xy, product_r_spec)
    else:
        r_pad_divisor = 1
        out_r = out_y

    logical_extent = r_end - r_start
    from runtime.padding import padded_axis
    r_axis = padded_axis(
        logical_extent, r_pad_divisor, name="WFN real-space chunk carrier")
    carrier_extent = r_axis.carrier
    if (product_reshard is not None and carrier_extent != logical_extent
            and r_end != n_rtot):
        raise ValueError(
            "prepare_rchunk_carrier: product-r padding is only inert on "
            "a terminal slab ending at the physical FFT-grid extent; got "
            f"[{r_start}, {r_end}) of [0, {n_rtot})")

    def _finish(a):
        if int(a.shape[-1]) != carrier_extent:
            raise AssertionError(
                "prepare_rchunk_carrier: canonical padded-r slice returned "
                f"extent {int(a.shape[-1])}, expected runtime.padding "
                f"carrier {carrier_extent}")
        if product_reshard is None:
            return jax.lax.with_sharding_constraint(a, out_y)
        return product_reshard(a)

    return r_axis, out_r, _finish


def iter_psi_rchunk_bandwise(
    wfn, sym, meta, mesh_xy, band_range, r_start, r_end, bispinor,
    band_chunk_size: int = 16,
    k_chunk_size: int = 0,
    band_chunk_ranges: list[tuple[int, int]] | None = None,
    band_pad_to: int | None = None,
    product_r_spec: P | None = None,
    bispinor_lift: str = "raw",
):
    """Generator: yield ``(bc_range, psi_bc_r)`` one band chunk at a time; see docs/architecture/zeta_fit_face_psi_cct.md."""
    del sym

    b_start, b_end = band_range
    nk_tot = int(meta.nk_tot)
    nk_batch = nk_tot if k_chunk_size <= 0 else min(k_chunk_size, nk_tot)
    n_rchunk = int(r_end - r_start)
    r_axis, out_r, _finish_r_carrier = prepare_rchunk_carrier(
        mesh_xy,
        r_start=r_start,
        r_end=r_end,
        n_rtot=meta.n_rtot,
        product_r_spec=product_r_spec,
    )
    n_rchunk_carrier = r_axis.carrier

    if band_chunk_ranges is None:
        nb_total = b_end - b_start
        num_band_chunks = (nb_total + band_chunk_size - 1) // band_chunk_size
        band_chunk_ranges = [
            (b_start + i * band_chunk_size,
             min(b_start + (i + 1) * band_chunk_size, b_end))
            for i in range(num_band_chunks)
        ]

    # JIT'd zero-allocator used by the k-chunked path (memoised by
    # shape).  Keeps the top-level ``jnp.zeros`` from being
    # rematerialised replicated on every device.
    _zeros_out_cache: dict = {}
    def _zeros_out(shape):
        fn = _zeros_out_cache.get(shape)
        if fn is None:
            fn = jax.jit(
                lambda: jnp.zeros(shape, dtype=jnp.complex128),
                out_shardings=out_r)
            _zeros_out_cache[shape] = fn
        return fn()

    sharding_load = band_sphere_spec()

    loader = wfn  # reuse top-level WfnLoader
    # Reuse the loader-owned replicated device table.  Passing the host table
    # here would make ``to_rchunk`` allocate a second identical G-vector to
    # FFT-box map beside the centroid path's canonical buffer.
    g_index_full = loader.box_index_dev(k="full_bz", mesh=mesh_xy)
    # WfnLoader owns the paired (k,G) gauge. A local integer-grid rebuild can
    # choose k+G_lattice while ``box_index`` still labels coefficients for k,
    # corrupting the full-Bloch phase.
    kvecs_frac_full = loader.kvecs(k="full_bz")

    for bc_range in band_chunk_ranges:
        if nk_batch >= nk_tot:
            # Shared capped-load + uniform-band-pad dance (zero-fill the
            # band axis to ``band_pad_to`` so ``to_rchunk`` sees ONE
            # shape across every chunk; trailing zero bands survive the
            # FFT + r-slice as zeros and are dropped by the caller's
            # zero-padded UH slice).
            psi_G_bc = load_psi_gflat_padded(
                loader, bc_range, mesh_xy=mesh_xy, bispinor=bispinor,
                pad_to=band_pad_to, k="full_bz", sharding=sharding_load,
                bispinor_lift=bispinor_lift)
            if psi_G_bc is None:
                raise ValueError(
                    f"iter_psi_rchunk_bandwise: band chunk {bc_range} lies "
                    f"entirely past the file's band extent "
                    f"({int(loader.nbands)})")
            psi_bc_r = to_rchunk(
                psi_G_bc, g_index_full, meta.fft_grid,
                int(r_start), n_rchunk_carrier, mesh=mesh_xy, norm="ortho",
                kvecs_frac=jnp.asarray(kvecs_frac_full),
                allow_padded_tail=(n_rchunk_carrier > n_rchunk))
            psi_bc_r = _finish_r_carrier(psi_bc_r)
            del psi_G_bc
            yield bc_range, psi_bc_r
        else:
            nb_chunk = bc_range[1] - bc_range[0]
            nspinor = meta.nspinor
            psi_bc_r_full = _zeros_out(
                (nk_tot, nb_chunk, nspinor, n_rchunk_carrier))
            for k0 in range(0, nk_tot, nk_batch):
                k1 = min(k0 + nk_batch, nk_tot)
                k_ids = list(range(k0, k1))
                psi_G_flat = loader.load(
                    bands=bc_range, k=k_ids,
                    sharding=sharding_load, bispinor=bispinor,
                    bispinor_lift=bispinor_lift)
                psi_k_chunk = to_rchunk(
                    psi_G_flat,
                    g_index_full[k0:k1],
                    meta.fft_grid, int(r_start), n_rchunk_carrier,
                    mesh=mesh_xy, norm="ortho",
                    kvecs_frac=jnp.asarray(kvecs_frac_full[k0:k1]),
                    allow_padded_tail=(n_rchunk_carrier > n_rchunk))
                psi_k_chunk = _finish_r_carrier(psi_k_chunk)
                psi_bc_r_full = psi_bc_r_full.at[
                    k0:k1, :, :, :].set(psi_k_chunk)
                del psi_G_flat, psi_k_chunk
            yield bc_range, psi_bc_r_full


# ============================================================================
# Unified band-chunked FFT backend for centroid and z-chunk extraction
# ============================================================================


def load_centroids_band_chunked(
    wfn,
    sym,
    meta: "Meta",
    centroid_indices: jax.Array,
    bispinor: bool,
    mesh_xy: Mesh,
    band_range: tuple[int, int],
    band_chunk_size: int = 64,
    k_chunk_size: int | None = None,
    *,
    psi_G_flat: jax.Array | None = None,
    bispinor_lift: str = "raw",
    k_domain: str = "full_bz",
    return_ibz_parents: bool = False,
) -> tuple[jax.Array, ...]:
    """Load centroid-sampled wavefunctions using band AND k-point chunking; see docs/architecture/zeta_fit_face_psi_cct.md."""
    del sym        # WfnLoader builds its own SymMaps lazily
    from common import timing
    from runtime.padding import padded_mu_extent

    b_start, b_end = band_range
    nb_total = b_end - b_start
    domain = str(k_domain).strip().lower()
    if domain not in ("full_bz", "ibz"):
        raise ValueError(
            "load_centroids_band_chunked: k_domain must be 'full_bz' or "
            f"'ibz'; got {k_domain!r}.")
    if domain == "ibz" and psi_G_flat is not None:
        raise ValueError(
            "load_centroids_band_chunked: psi_G_flat is a full-BZ reuse "
            "carrier and cannot be paired with k_domain='ibz'.")
    return_parents = bool(return_ibz_parents)
    if return_parents and domain != "full_bz":
        raise ValueError(
            "load_centroids_band_chunked: return_ibz_parents requires "
            "k_domain='full_bz'.")
    if return_parents and psi_G_flat is not None:
        raise ValueError(
            "load_centroids_band_chunked: return_ibz_parents needs the "
            "parent-major loader stream, not a pre-unfolded psi_G_flat.")
    nk_tot = (int(meta.nk_tot) if domain == "full_bz"
              else int(wfn.nkpts))
    nspinor = int(meta.nspinor)
    # The run's in-memory centroid order (``meta.mu_basis``): sample ψ at the
    # PACKED table and zero its pad slots, so every face leaves here in the
    # order the whole run computes in.  Without a basis the canonical table
    # is sampled and suffix-padded as before.
    mu_basis = getattr(meta, 'mu_basis', None)
    mu_active_mask = None
    if mu_basis is not None:
        centroid_indices = mu_basis.packed_indices
        mu_active_mask = np.asarray(mu_basis.active_mask, dtype=bool)
    n_rmu = int(centroid_indices.shape[0])
    centroid_idx_np = np.asarray(centroid_indices, dtype=np.int32)
    n_rtot = int(meta.fft_grid[0]) * int(meta.fft_grid[1]) * int(meta.fft_grid[2])

    # Defect 3 (zeta_rchunk_memory_model_2026-05-13/defect_catalog.md):
    # the legacy bc-loop here, paired with the unsharded FFT box inside
    # ``to_rmu``, materialised an unsharded ``c128[nk, band_chunk, ns,
    # nx, ny, nz]`` transient on every rank — Peak A in
    # ``gw/gflat_memory_model.py``.  Single slot, but the §0
    # zero-replicated-intermediates principle still bites.  ``gflat_to_rmu``
    # fuses each fixed-shape band/k tile into one shard_map + lax.scan whose
    # per-iter FFT box is sharded along the band axis on ``('x','y')`` and
    # aliased across scan iters.  The two outer tile sizes bound the WFN and
    # transform operands; ``chunk_size`` below independently bounds FFT rows
    # within one tile.

    # Per-iter FFT box bound for ``gflat_to_rmu``: each scan iter holds
    # one ``c128[cs, ns, nx, ny, nz]`` box per rank.  ``peak_copies``
    # is the same conservative XLA scratch multiplier used historically
    # by the old k_chunk_size autodetect (4 on single-rank, 9 on
    # multi-rank — covers the IFFT scratch + IFFT output).
    sharding_load = band_sphere_spec()
    p_band = spec_divisor(mesh_xy, sharding_load, axis=1)
    mesh_devices = int(mesh_xy.size)
    peak_copies = 4 if mesh_devices == 1 else 9
    gpu_mem_bytes = 36e9
    if hasattr(meta, 'memory_per_device_gb') and meta.memory_per_device_gb > 0:
        gpu_mem_bytes = meta.memory_per_device_gb * 1e9

    # Output shardings + accumulators.  Same final layout as before:
    # psi_rmu_Y has the centroid axis on 'y'; psi_rmuT_X has it on 'x'.
    out_Y = NamedSharding(mesh_xy, P(None, None, None, 'y'))
    out_X = NamedSharding(mesh_xy, P(None, 'x', None, None))
    stage_Y_4d = NamedSharding(mesh_xy, P(None, 'y', None, None))
    stage_X_4d = NamedSharding(mesh_xy, P(None, 'x', None, None))

    # A packed basis already owns its runtime extent. The extra-padding
    # knob applies only to canonical staging, never a second time here.
    n_rmu_padded = (mu_basis.n_packed if mu_basis is not None
                    else padded_mu_extent(n_rmu, mesh_xy))
    loader = wfn  # reuse top-level WfnLoader
    # The shared GW memory plan already owns a positive Stage-A band chunk;
    # honor it here instead of bulk-loading the very tensor it prices.  Prune
    # additionally supplies a positive k chunk.  A preloaded htransform
    # window deliberately stays bulk because its next Galerkin sweep reuses
    # that allocation.  Keep one fixed band/k tile shape (runtime-padding
    # both remainders) so the transform compiles once per physical window.
    k_stream_requested = (
        k_chunk_size is not None and int(k_chunk_size) > 0
    )
    requested_k_tile = (
        1 if return_parents else
        (int(k_chunk_size) if k_stream_requested else nk_tot)
    )
    requested_band_tile = min(
        nb_total, max(1, int(band_chunk_size)))
    stream_tiles = (
        psi_G_flat is None
        and (return_parents or k_stream_requested
             or requested_band_tile < nb_total)
    )
    k_tile = min(nk_tot, requested_k_tile) if stream_tiles else nk_tot
    band_tile = nb_total
    if stream_tiles:
        from runtime.padding import padded_axis
        band_tile = padded_axis(
            requested_band_tile, p_band,
            name="centroid stream band tile").carrier
    from runtime.padding import padded_axis
    nk_accum = padded_axis(
        nk_tot, k_tile, name="centroid stream k accumulator").carrier
    nb_accum = padded_axis(
        nb_total, band_tile, name="centroid stream band accumulator").carrier
    parent_groups = None
    parent_stream_active = False
    from wfn_loader import IBZRows
    if domain == "full_bz" and stream_tiles and k_tile == 1:
        # The loader owns canonical parent/star membership and child order;
        # this consumer never reads or re-groups raw SymMaps tables.
        parent_groups = loader.full_k_parent_groups()
        parent_stream_active = True
    max_parent_star = (
        max(len(children) for _, children in parent_groups)
        if parent_groups else 0)

    # Persistent term for the transfer. Bulk retains the complete sharded
    # G-flat input. Streaming retains one fixed child tile, one raw parent,
    # and at most one physical symmetry star of int32 FFT indices beside the
    # donated final X/Y accumulators. Tile sample/reshard faces are priced too
    # because they coexist at the insert boundary.
    nb_per_band_shard = (
        band_tile // p_band if stream_tiles
        else (
            padded_axis(
                nb_total, p_band,
                name="centroid bulk band carrier").carrier
            // p_band)
    )
    nb_padded_global = (
        band_tile if stream_tiles else nb_per_band_shard * p_band
    )
    if not stream_tiles:
        nb_accum = nb_padded_global
    n_x = int(mesh_xy.shape['x']) if 'x' in mesh_xy.axis_names else 1
    n_y = int(mesh_xy.shape['y']) if 'y' in mesh_xy.axis_names else 1
    gflat_local_bytes = (
        k_tile * nb_per_band_shard * nspinor * int(loader.ngkmax) * 16
    )
    # A k=1 parent/star schedule retains one raw file-spinor parent, its
    # integer G row, and the current physical star's FFT gather indices.
    # It is bounded by the symmetry-group order, not nk; these carriers coexist
    # at the FFT boundary
    # and therefore belong in the refusal model.  In 4c mode the unfolded 2c
    # child also coexists during the post-unfold lift, so price that
    # short-lived carrier as well.
    parent_stream_extra_bytes = 0
    if parent_stream_active:
        one_raw_2c_local_bytes = (
            nb_per_band_shard * int(loader.nspinor)
            * int(loader.ngkmax) * 16
        )
        parent_stream_extra_bytes = one_raw_2c_local_bytes * (
            2 if bispinor else 1)
        parent_stream_extra_bytes += int(loader.ngkmax) * 3 * 4
        # One-child FFT-index construction is a functional scatter. Price one
        # star of outputs plus a deliberately conservative live temporary set:
        # rotated
        # G/components/cells/mask/update indices and a scatter output copy.
        # All terms remain bounded by one symmetry star, never O(nk).
        parent_stream_extra_bytes += (
            int(loader.ngkmax) * 40
            + max_parent_star * n_rtot * 4
            + n_rtot * 4)
        if bispinor:
            # The fused child-G transform/kinetic-balance lift needs one
            # float64 Cartesian momentum row. It is device-local and never
            # enters the loader's host full-k G cache.
            parent_stream_extra_bytes += int(loader.ngkmax) * 3 * 8
    from runtime.padding import padded_axis
    mu_x_local = padded_axis(
        n_rmu_padded, n_x,
        name="centroid X-shard memory carrier").carrier // n_x
    mu_y_local = padded_axis(
        n_rmu_padded, n_y,
        name="centroid Y-shard memory carrier").carrier // n_y
    output_local_bytes = (
        nk_accum * nb_accum * nspinor * 16
        * (mu_x_local + mu_y_local)
    )
    if return_parents:
        output_local_bytes += (
            int(loader.nkpts) * nb_accum * nspinor * 16
            * (mu_x_local + mu_y_local)
        )
    tile_band_output_local_bytes = 0
    tile_face_local_bytes = 0
    if stream_tiles:
        tile_band_output_local_bytes = (
            k_tile * nb_per_band_shard * nspinor * n_rmu * 16
        )
        tile_face_local_bytes = (
            k_tile * band_tile * nspinor * 16
            * (mu_x_local + mu_y_local)
        )
    # A caller-provided G-flat tensor is already in memory_stats(); only price
    # it here when this call will allocate it.  This avoids double-charging
    # htransform's shared-window reuse path.
    # Every loader unfold has a request-local c128 nonsymmorphic phase
    # temporary.  Parent streaming generates it inside the device action from
    # the one cached integer G row above; other routes stage the same logical
    # rows.  Charge the maximum simultaneous rows alongside G-flat data.
    phase_rows = k_tile if stream_tiles else nk_tot
    request_phase_bytes = phase_rows * int(loader.ngkmax) * 16
    new_gflat_bytes = (
        gflat_local_bytes + parent_stream_extra_bytes + request_phase_bytes
        if psi_G_flat is None else 0
    )
    persistent_bytes = (
        new_gflat_bytes + output_local_bytes
        + tile_band_output_local_bytes + tile_face_local_bytes
    )
    min_scan_bytes = nspinor * n_rtot * 16 * peak_copies
    existing_live_local_bytes = 0
    for device in jax.local_devices():
        stats = device.memory_stats() or {}
        existing_live_local_bytes = max(
            existing_live_local_bytes, int(stats.get("bytes_in_use") or 0),
        )
    # ``cs`` below is a static scan shape and cache-key component.  Allocator
    # residency can differ by process after asynchronous Lloyd/JIT teardown;
    # use one shared worst-rank floor so every process compiles the same scan.
    existing_live_bytes = worst_process_resident_bytes(
        existing_live_local_bytes)
    scan_budget_bytes = (int(gpu_mem_bytes) - existing_live_bytes
                         - persistent_bytes)
    if scan_budget_bytes < min_scan_bytes:
        min_live_bytes = (existing_live_bytes + persistent_bytes
                          + min_scan_bytes)
        raise MemoryError(
            "load_centroids_band_chunked planner refuses before WFN "
            f"allocation: the minimum per-device live set is "
            f"{min_live_bytes / 2**30:.2f} GiB (G-flat "
            f"{'tile' if stream_tiles else 'input'} + X/Y centroid "
            f"outputs + one FFT scan row), but the residual prune "
            f"transient budget is {gpu_mem_bytes / 2**30:.2f} GiB. "
            "A smaller scan chunk cannot reduce this floor; use more "
            "devices, larger-HBM devices, or a narrower prune band window."
        )

    # Translate legacy hints (band_chunk_size, k_chunk_size) into the new
    # flat-row count.  Only the budget LEFT AFTER persistent arrays may size
    # the scan transient.  In either arm the bound applies last.
    cs_budget = max(1, int(
        scan_budget_bytes // (nspinor * n_rtot * 16 * peak_copies)
    ))
    if stream_tiles:
        cs_hint = k_tile * nb_per_band_shard
        cs = max(1, min(cs_hint, cs_budget))
    else:
        cs = cs_budget
    print(
        "[load_centroids planner] "
        f"existing={existing_live_bytes / 2**30:.2f}, "
        f"persistent={persistent_bytes / 2**30:.2f}, "
        f"scan_budget={scan_budget_bytes / 2**30:.2f} GiB/device, "
        f"peak_copies={peak_copies}, cs={cs}, "
        f"stream={'on' if stream_tiles else 'off'}, "
        f"k_domain={domain}, retain_parents={'yes' if return_parents else 'no'}, "
        f"k_tile={k_tile}, band_tile={band_tile}"
    )
    # The one-k parent schedule builds each current-star child index from the
    # resident parent G row. Other streaming/bulk schedules retain the
    # established complete cached table.
    g_index_full = None
    if parent_groups is None:
        g_index_full = loader.box_index_dev(k=domain, mesh=mesh_xy)
    # Use the k representatives paired with this loader's typed full-BZ map.
    kvecs_frac_full = loader.kvecs(k=domain)

    # One reshard owner for both the bulk and streamed paths.  The input is
    # band-sharded; the two consumers need independent Y-face and transposed
    # X-face layouts.  Applying each stage constraint before μ padding avoids
    # the x-major/y-major involuntary rematerialization documented below.
    @partial(jax.jit, out_shardings=(out_Y, out_X))
    def _reshard_centroid_tile(psi_rmu_band):
        pad_cfg = ((0, 0), (0, 0), (0, 0), (0, n_rmu_padded - n_rmu))
        if mu_active_mask is not None:
            psi_rmu_band = jnp.where(
                jnp.asarray(mu_active_mask)[None, None, None, :],
                psi_rmu_band, jnp.zeros((), dtype=psi_rmu_band.dtype))
        psi_rmu = jax.lax.with_sharding_constraint(psi_rmu_band, stage_Y_4d)
        if n_rmu_padded > n_rmu:
            psi_rmu = jnp.pad(psi_rmu, pad_cfg)
        psi_rmu = jax.lax.with_sharding_constraint(psi_rmu, out_Y)
        psi_T = jax.lax.with_sharding_constraint(psi_rmu_band, stage_X_4d)
        if n_rmu_padded > n_rmu:
            psi_T = jnp.pad(psi_T, pad_cfg)
        psi_rmuT = jnp.conj(psi_T.transpose(0, 3, 1, 2))
        psi_rmuT = jax.lax.with_sharding_constraint(psi_rmuT, out_X)
        return psi_rmu, psi_rmuT

    def _finish_faces(psi_rmu_all, psi_rmuT_all, nk_result):
        # Remove fixed-shape stream/loader pad rows before returning the
        # public logical k and band extents.
        nk_result = int(nk_result)
        if int(psi_rmu_all.shape[0]) > nk_result:
            psi_rmu_all = psi_rmu_all[:nk_result]
            psi_rmuT_all = psi_rmuT_all[:nk_result]
        if int(psi_rmu_all.shape[1]) > nb_total:
            psi_rmu_all = psi_rmu_all[:, :nb_total, :, :]
            psi_rmuT_all = psi_rmuT_all[:, :, :nb_total, :]
        psi_rmu_all = jax.lax.with_sharding_constraint(psi_rmu_all, out_Y)
        psi_rmuT_all = jax.lax.with_sharding_constraint(psi_rmuT_all, out_X)

        # Zero user-band-pad rows (unchanged contract).
        nb_user_in_range = max(0, meta.b_id_4_user - b_start)
        if nb_user_in_range < nb_total:
            zero_y = jnp.zeros_like(
                psi_rmu_all[:, nb_user_in_range:nb_total, :, :])
            zero_x = jnp.zeros_like(
                psi_rmuT_all[:, :, nb_user_in_range:nb_total, :])
            psi_rmu_all = psi_rmu_all.at[
                :, nb_user_in_range:nb_total, :, :].set(zero_y)
            psi_rmuT_all = psi_rmuT_all.at[
                :, :, nb_user_in_range:nb_total, :].set(zero_x)
        return psi_rmu_all, psi_rmuT_all

    if b_start >= int(loader.nbands):
        raise ValueError(
            "load_centroids_band_chunked: band window "
            f"({b_start}, {b_end}) lies entirely past the file's "
            f"band extent ({int(loader.nbands)})")

    if stream_tiles:
        @partial(jax.jit, out_shardings=(out_Y, out_X))
        def _zero_faces():
            return (
                jnp.zeros(
                    (nk_accum, nb_accum, nspinor, n_rmu_padded),
                    dtype=jnp.complex128),
                jnp.zeros(
                    (nk_accum, n_rmu_padded, nb_accum, nspinor),
                    dtype=jnp.complex128),
            )

        @partial(jax.jit, out_shardings=(out_Y, out_X))
        def _zero_parent_faces():
            return (
                jnp.zeros(
                    (int(loader.nkpts), nb_accum, nspinor, n_rmu_padded),
                    dtype=jnp.complex128),
                jnp.zeros(
                    (int(loader.nkpts), n_rmu_padded, nb_accum, nspinor),
                    dtype=jnp.complex128),
            )

        @partial(
            jax.jit, donate_argnums=(0, 1), out_shardings=(out_Y, out_X))
        def _insert_tile(acc_y, acc_x, psi_rmu_band, k0, b0):
            tile_y, tile_x = _reshard_centroid_tile(psi_rmu_band)
            zero = jnp.int32(0)
            acc_y = jax.lax.dynamic_update_slice(
                acc_y, tile_y, (k0, b0, zero, zero))
            acc_x = jax.lax.dynamic_update_slice(
                acc_x, tile_x, (k0, zero, b0, zero))
            return acc_y, acc_x

        def _sample_and_insert_one(
                acc_y, acc_x, psi_G_one, g_index_one, output_row, b_rel,
                kvecs_one):
            kvecs_one = jnp.asarray(kvecs_one).reshape(1, 3)
            with timing.section("load_centroids.gflat_to_rmu"):
                psi_rmu_band = gflat_to_rmu(
                    psi_G_one, g_index_one, centroid_idx_np,
                    mesh=mesh_xy, fft_grid=meta.fft_grid,
                    kvecs_frac=kvecs_one, norm="ortho", chunk_size=cs)
                jax.block_until_ready(psi_rmu_band)
            del kvecs_one

            with timing.section("load_centroids.reshard_insert"):
                acc_y, acc_x = _insert_tile(
                    acc_y, acc_x, psi_rmu_band,
                    jnp.int32(output_row), jnp.int32(b_rel))
                jax.block_until_ready((acc_y, acc_x))
            del psi_rmu_band
            return acc_y, acc_x

        psi_rmu_all, psi_rmuT_all = _zero_faces()
        psi_parent_y = psi_parent_x = None
        if return_parents:
            psi_parent_y, psi_parent_x = _zero_parent_faces()
            kvecs_parent = loader.kvecs(k="ibz")

        if parent_groups is not None:
            # Keep each parent's integer G row resident across all of its
            # band tiles and children.  This avoids rebuilding or transferring
            # an ngk-sized phase row per child while retaining the established
            # one-full-k transform/IFFT workspace. Every group, including a
            # singleton, uses the same raw-parent door. Its star of child
            # indices is reused across band tiles, then released before the
            # next parent, so neither G vectors nor indices accumulate with nk.
            for parent, full_children in parent_groups:
                with timing.section("load_centroids.parent_box_indices"):
                    parent_index = (
                        loader.ibz_box_index_one_dev(int(parent))
                        if return_parents else None)
                    child_indices = [
                        (int(child), loader.full_k_box_index_one_dev(
                            int(child)))
                        for child in full_children
                    ]
                    ready_indices = tuple(
                        index for _, index in child_indices)
                    if parent_index is not None:
                        ready_indices += (parent_index,)
                    jax.block_until_ready(ready_indices)
                for b_rel in range(0, nb_total, band_tile):
                    b_hi_rel = min(b_rel + band_tile, nb_total)
                    band_window = (b_start + b_rel, b_start + b_hi_rel)
                    with timing.section("load_centroids.loader_load"):
                        parent_psi = load_psi_gflat_padded(
                            loader, band_window, mesh_xy=mesh_xy,
                            bispinor=False, pad_to=band_tile,
                            k=IBZRows((int(parent),)),
                            sharding=sharding_load,
                            bispinor_lift="raw")
                        # A terminal band tile can lie wholly beyond mnband.
                        if parent_psi is None:
                            continue
                        jax.block_until_ready(parent_psi)

                    if return_parents:
                        retained_parent = parent_psi
                        if bispinor:
                            retained_parent = load_psi_gflat_padded(
                                loader, band_window, mesh_xy=mesh_xy,
                                bispinor=True, pad_to=band_tile,
                                k=IBZRows((int(parent),)),
                                sharding=sharding_load,
                                bispinor_lift=bispinor_lift)
                        psi_parent_y, psi_parent_x = _sample_and_insert_one(
                            psi_parent_y, psi_parent_x, retained_parent,
                            parent_index, int(parent), b_rel,
                            kvecs_parent[int(parent)])
                        del retained_parent

                    for child, g_index_one in child_indices:
                        with timing.section("load_centroids.parent_unfold"):
                            psi_G_tile = loader.unfold_parent_to_full_k(
                                parent_psi, parent=int(parent), full_k=child,
                                bispinor=bispinor,
                                bispinor_lift=bispinor_lift)
                            jax.block_until_ready(psi_G_tile)

                        psi_rmu_all, psi_rmuT_all = _sample_and_insert_one(
                            psi_rmu_all, psi_rmuT_all, psi_G_tile, g_index_one,
                            child, b_rel, kvecs_frac_full[child])
                        del psi_G_tile
                    del parent_psi
                del child_indices

        else:
            for b_rel in range(0, nb_total, band_tile):
                b_hi_rel = min(b_rel + band_tile, nb_total)
                band_window = (b_start + b_rel, b_start + b_hi_rel)

                for k0 in range(0, nk_tot, k_tile):
                    k1 = min(k0 + k_tile, nk_tot)
                    k_ids = list(range(k0, k1))
                    with timing.section("load_centroids.loader_load"):
                        psi_G_tile = load_psi_gflat_padded(
                            loader, band_window, mesh_xy=mesh_xy,
                            bispinor=bispinor, pad_to=band_tile,
                            k=(k_ids if domain == "full_bz"
                               else IBZRows(tuple(k_ids))),
                            sharding=sharding_load,
                            bispinor_lift=bispinor_lift)
                        # A terminal band tile can lie wholly beyond mnband
                        # when Meta rounded the user's logical edge to the
                        # mesh.  The accumulator is already exact zero there.
                        if psi_G_tile is None:
                            continue
                        psi_G_tile = pad_axis(
                            psi_G_tile, k_tile, axis=0).array
                        jax.block_until_ready(psi_G_tile)

                    g_index_tile = pad_axis(
                        g_index_full[k0:k1], k_tile, axis=0).array
                    kvecs_tile = pad_axis(
                        jnp.asarray(kvecs_frac_full[k0:k1]),
                        k_tile, axis=0).array
                    with timing.section("load_centroids.gflat_to_rmu"):
                        psi_rmu_band = gflat_to_rmu(
                            psi_G_tile, g_index_tile, centroid_idx_np,
                            mesh=mesh_xy, fft_grid=meta.fft_grid,
                            kvecs_frac=kvecs_tile, norm="ortho",
                            chunk_size=cs)
                        jax.block_until_ready(psi_rmu_band)
                    del psi_G_tile, g_index_tile, kvecs_tile

                    with timing.section("load_centroids.reshard_insert"):
                        psi_rmu_all, psi_rmuT_all = _insert_tile(
                            psi_rmu_all, psi_rmuT_all, psi_rmu_band,
                            jnp.int32(k0), jnp.int32(b_rel))
                        jax.block_until_ready((psi_rmu_all, psi_rmuT_all))
                    del psi_rmu_band

        gc.collect()
        full_faces = _finish_faces(psi_rmu_all, psi_rmuT_all, nk_tot)
        if not return_parents:
            return full_faces
        parent_faces = _finish_faces(
            psi_parent_y, psi_parent_x, int(loader.nkpts))
        return (*full_faces, *parent_faces)

    # Pull all (nk_tot, nb_padded, ns, ngkmax) ψ(G-flat) onto device in
    # one collective load.  The G-flat tensor is small relative to the
    # FFT box (n_rmu << n_rtot, ngkmax << n_rtot too once the G-sphere
    # cutoff is applied) so a single-shot load fits comfortably even at
    # CrI3 6×6 80 Ry scale (~50 GB total / mesh.size).  The band-pad
    # padding happens inside ``loader.load`` so ``nb_padded`` is the
    # mesh-aligned extent expected by ``gflat_to_rmu``.
    #
    # Past-mnband zero-pad (the contract promised by Meta docstring at
    # ``common/meta.py:100-117``): ``b_id_4 = _round_up(b_id_4_user,
    # world_size)`` may exceed the file's ``mnband`` whenever
    # ``world_size`` rounds the user's band count past the file extent
    # (CrI3 6×6 30Ry SOC: mnband=86, world_size=16 ⇒ b_id_4=96).
    # ``WfnLoader.load`` validates ``b_hi <= self.nbands`` at
    # ``file_io/wfn_loader.py:678``; without the cap below every rank
    # raises at module init.  Cap the loader call at ``loader.nbands``
    # and zero-pad the band axis up to ``nb_total = b_end - b_start``,
    # preserving the (None, ('x','y'), None, None) sharding.  The pad
    # rows are physically zero — same contract the user-band-pad zero
    # block below applies to the (b_id_4_user, b_id_4) slots.
    # ``psi_G_flat`` may arrive pre-loaded from the caller (htransform
    # loads ONE window that serves both this centroid sample and the
    # r-streaming sweep); otherwise pull it via the shared helper.
    if psi_G_flat is None:
        with timing.section("load_centroids.loader_load"):
            psi_G_flat = load_psi_gflat_padded(
                loader, (b_start, b_end), mesh_xy=mesh_xy,
                bispinor=bispinor, k=domain, sharding=sharding_load,
                bispinor_lift=bispinor_lift)
            if psi_G_flat is None:
                raise ValueError(
                    f"load_centroids_band_chunked: band window "
                    f"({b_start}, {b_end}) lies entirely past the file's "
                    f"band extent ({int(loader.nbands)})")
            jax.block_until_ready(psi_G_flat)
    # One shard_map + scan: full (nk, nb_padded) extent, centroid
    # samples emitted band-sharded.  FFT box exists once at scan-body
    # scope and is per-rank-local (mesh.size× smaller than the legacy
    # unsharded transient).
    with timing.section("load_centroids.gflat_to_rmu"):
        psi_rmu_band = gflat_to_rmu(
            psi_G_flat, g_index_full, centroid_idx_np,
            mesh=mesh_xy, fft_grid=meta.fft_grid,
            kvecs_frac=jnp.asarray(kvecs_frac_full),
            norm="ortho", chunk_size=cs)
        jax.block_until_ready(psi_rmu_band)
    del psi_G_flat

    # Single global reshard {None, XY, None, None} → {None, None, None, Y}
    # plus a conjugate-transpose into the rmuT_X layout.  TWO INDEPENDENT
    # staged chains from the band-sharded input, one per output:
    #
    #   psi_rmu :  b:('x','y') → b:'y'  → μ:'y'   (out_Y)
    #   psi_rmuT:  b:('x','y') → b:'x'  → transpose → μ:'x'  (out_X)
    #
    # Each chain is one same-axis band→μ all-to-all after a partial
    # gather over the OTHER axis — both transitions XLA's partitioner
    # handles natively.  The previous form derived psi_rmuT from the
    # FINISHED out_Y tensor, which asked for a μ:'y' → μ:'x' reshard on
    # the transposed tensor; that is the x-major↔y-major device-order
    # move XLA cannot lower (upstream b/433785288) and it compiled into
    # a compiler-flagged "[SPMD] Involuntary full rematerialization" —
    # an all-gather of the ENTIRE (nk, μ_pad, nb, ns) ψ_rμ on every rank
    # (1.475 GB/rank at 1998c/nb160/P=80; grows ∝ μ·nb — scorecard K.2,
    # K.1 #7).  The split chains move full/Px + full/Py per rank instead
    # of full/Py + full.  Values are untouched (pad/transpose/conj are
    # elementwise/layout ops): bit-identical outputs, different wires.
    # ORDERING MATTERS: each chain applies its stage constraint BEFORE the
    # μ-pad.  Padding first (the original form) gives the pad output ONE
    # sharding that both chains then consume — the partitioner assigns it
    # the first chain's b:'y' layout and the second chain's b:'x' demand
    # becomes exactly the y-major↔x-major device-order move again
    # (measured: at 8×10 the hoisted pad re-created the involuntary remat
    # on per-shard c128[144,16,2,640] — wk_AO/gw80 first attempt).  With
    # the constraint first, the two pads are distinct instructions on
    # distinctly-sharded operands and each chain stays on its own axis.
    # Attribution barrier: the process-local WfnLoader work above this line
    # does not synchronize processes, so the first collective inside
    # ``_reshard_centroid_tile`` absorbs ALL process skew accumulated since
    # launch
    # (cold-start import/startup skew).  Measured: the identical
    # 606c/P=80 cell pair in job 7876541 recorded reshard = 92.6 s on
    # the allocation's first (cold) srun vs 0.55 s on the third run on
    # the SAME nodes with the same 433 compiles — the 92 s was never
    # data movement.  This named barrier charges the skew to its own
    # row so ``load_centroids.reshard`` reports the collective itself.
    with timing.section("load_centroids.pre_reshard_sync"):
        from common.collectives import barrier as _sync_barrier
        _sync_barrier("load_centroids_pre_reshard")

    with timing.section("load_centroids.reshard"):
        psi_rmu_all, psi_rmuT_all = _reshard_centroid_tile(psi_rmu_band)
        jax.block_until_ready(psi_rmuT_all)
    del psi_rmu_band
    gc.collect()
    return _finish_faces(psi_rmu_all, psi_rmuT_all, nk_tot)
