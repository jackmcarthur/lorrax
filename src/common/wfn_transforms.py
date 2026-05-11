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


__all__ = ["to_box", "to_rbox", "to_rmu", "to_rchunk", "apply_bloch_phase"]


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
    *,
    norm: str = "backward",
    kvecs_frac: np.ndarray | jax.Array | None = None,
) -> jax.Array:
    """Scatter ψ to the FFT box and IFFT to r-space.

    Same shape as :func:`to_box`; the trailing three axes are now real-
    space.  The IFFT is over axes ``(-3, -2, -1)``; ``norm`` is forwarded
    to :func:`jnp.fft.ifftn` (``'backward'`` matches ``np.fft.ifftn``'s
    default = ``1/N``; ``'ortho'`` = ``1/√N`` on both directions, used
    by the centroid pivoted-Cholesky path).

    Optional ``kvecs_frac`` (n_k, 3) applies the Bloch phase
    ``exp(2πi k·r)`` after the IFFT, converting the periodic part
    ``u_nk(r)`` = IFFT(c_nk(G)) into the full Bloch state
    ``ψ_nk(r) = exp(2πi k·r) · u_nk(r)`` — matches the convention
    used everywhere ψ_r is the downstream consumer (ISDF r-chunk fit,
    kin_ion matrix elements).  Default ``None`` skips the phase
    (e.g. when only ``|ψ|²`` is wanted as in centroid charge density).

    Memory: this still materialises the FFT box.  For consumers that
    only need ψ at a centroid list or a flat-r slab, prefer
    :func:`to_rmu` / :func:`to_rchunk`.
    """
    psi_box = to_box(psi, g_index, fft_grid)
    psi_r_box = jnp.fft.ifftn(psi_box, axes=(-3, -2, -1), norm=norm)
    if kvecs_frac is not None:
        psi_r_box = apply_bloch_phase(
            psi_r_box,
            jnp.asarray(kvecs_frac, dtype=jnp.float64),
            tuple(int(s) for s in fft_grid))
    return psi_r_box


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

    Parameters
    ----------
    psi, g_index, fft_grid
        As :func:`to_box`.
    r_mu : (n_rmu, 3) int32
        Centroid positions as FFT-grid indices in ``[0, fft_grid[a])``.
    norm, kvecs_frac
        Forwarded to :func:`to_rbox`.

    Returns
    -------
    psi_at_rmu : (n_k, nb, nspinor, n_rmu) c128
        Band-axis sharding preserved.
    """
    psi_r_box = to_rbox(psi, g_index, fft_grid, norm=norm, kvecs_frac=kvecs_frac)
    r_mu_j = jnp.asarray(r_mu, dtype=jnp.int32)
    out = psi_r_box[:, :, :, r_mu_j[:, 0], r_mu_j[:, 1], r_mu_j[:, 2]]
    return _maybe_constrain(out, _output_sharding(psi, n_extra_axes=1))


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
    # r0 may be a Python int (driver-time check) or a jax scalar tracer
    # (when ``to_rchunk`` is called inside an outer jit).  Skip the
    # bounds check on traced values; caller is responsible.
    if isinstance(r0, (int, np.integer)):
        if r0 < 0 or r0 + r_len > n_rtot:
            raise ValueError(
                f"to_rchunk: [{r0}, {r0 + r_len}) out of [0, {n_rtot})")
    psi_r_box = to_rbox(psi, g_index, fft_grid, norm=norm,
                        kvecs_frac=kvecs_frac)
    psi_r_flat = psi_r_box.reshape(*psi_r_box.shape[:3], n_rtot)
    out = jax.lax.dynamic_slice_in_dim(psi_r_flat, r0, int(r_len), axis=-1)
    return _maybe_constrain(out, _output_sharding(psi, n_extra_axes=1))


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
