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


__all__ = ["to_box", "to_rbox", "to_rmu", "to_rchunk"]


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
    gathered = jnp.take(psi_flat, flat_index, axis=2)             # (nb, ns, n_k, nx, ny, nz)
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
# Public transforms
# ---------------------------------------------------------------------------

def to_box(
    psi: jax.Array,
    g_index: np.ndarray | jax.Array,
    fft_grid: Sequence[int],
) -> jax.Array:
    """Scatter G-flat ψ into the FFT box.

    Parameters
    ----------
    psi : (n_k, nb, nspinor, ngkmax) c128
        Output of :meth:`WfnLoader.load`.  Sharding (band axis on
        ``('x','y')`` or replicated) is preserved on the output.
    g_index : (n_k, nx, ny, nz) int32
        Output of :meth:`WfnLoader.box_index`.  Sentinel value ``ngkmax``
        flags empty FFT-box cells; gather produces zero there.
    fft_grid : (nx, ny, nz)
        Used only for shape validation (must equal g_index's spatial dims).

    Returns
    -------
    psi_box : (n_k, nb, nspinor, nx, ny, nz) c128
    """
    ngkmax = int(psi.shape[-1])
    g_index_j = jnp.asarray(g_index, dtype=jnp.int32)
    out = _box_kernel(psi, g_index_j, ngkmax=ngkmax)
    return _maybe_constrain(out, _output_sharding(psi, n_extra_axes=3))


def to_rbox(
    psi: jax.Array,
    g_index: np.ndarray | jax.Array,
    fft_grid: Sequence[int],
) -> jax.Array:
    """Scatter ψ to the FFT box and IFFT to r-space.

    Same shape as :func:`to_box`; the trailing three axes are now real-
    space.  The IFFT is over axes ``(-3, -2, -1)`` with ``norm='backward'``
    (matches ``np.fft.ifftn`` default; consistent with the rest of
    LORRAX where ``ψ_r = IFFT(ψ_G_box)``).

    Memory: this still materialises the FFT box.  For consumers that
    only need ψ at a centroid list or a flat-r slab, prefer
    :func:`to_rmu` / :func:`to_rchunk`.
    """
    psi_box = to_box(psi, g_index, fft_grid)
    return jnp.fft.ifftn(psi_box, axes=(-3, -2, -1))


def to_rmu(
    psi: jax.Array,
    g_index: np.ndarray | jax.Array,
    fft_grid: Sequence[int],
    r_mu: np.ndarray | jax.Array,
) -> jax.Array:
    """ψ in r-space at the centroid FFT-grid indices ``r_mu``.

    Parameters
    ----------
    psi, g_index, fft_grid
        As :func:`to_box`.
    r_mu : (n_rmu, 3) int32
        Centroid positions as FFT-grid indices in ``[0, fft_grid[a])``.

    Returns
    -------
    psi_at_rmu : (n_k, nb, nspinor, n_rmu) c128
        Band-axis sharding preserved.
    """
    psi_r_box = to_rbox(psi, g_index, fft_grid)
    r_mu_j = jnp.asarray(r_mu, dtype=jnp.int32)
    out = psi_r_box[:, :, :, r_mu_j[:, 0], r_mu_j[:, 1], r_mu_j[:, 2]]
    return _maybe_constrain(out, _output_sharding(psi, n_extra_axes=1))


def to_rchunk(
    psi: jax.Array,
    g_index: np.ndarray | jax.Array,
    fft_grid: Sequence[int],
    r0: int,
    r_len: int,
) -> jax.Array:
    """ψ in r-space on a contiguous flat-r slab ``[r0, r0 + r_len)``.

    Flat-r convention: ``r_flat = rx * ny * nz + ry * nz + rz`` (matches
    C-order reshape of ``(nx, ny, nz)``).  Used by the ISDF r-chunk
    loop where the consumer wants a strip of the full FFT box without
    materialising the rest.

    Returns
    -------
    psi_rchunk : (n_k, nb, nspinor, r_len) c128
        Band-axis sharding preserved.
    """
    nx, ny, nz = (int(s) for s in fft_grid)
    n_rtot = nx * ny * nz
    if r0 < 0 or r0 + r_len > n_rtot:
        raise ValueError(
            f"to_rchunk: [{r0}, {r0 + r_len}) out of [0, {n_rtot})")
    psi_r_box = to_rbox(psi, g_index, fft_grid)
    psi_r_flat = psi_r_box.reshape(*psi_r_box.shape[:3], n_rtot)
    out = jax.lax.dynamic_slice_in_dim(psi_r_flat, int(r0), int(r_len), axis=-1)
    return _maybe_constrain(out, _output_sharding(psi, n_extra_axes=1))
