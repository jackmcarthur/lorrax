"""Sparse-G → dense-FFT-box via precomputed index + gather.

See ``GVEC_FFT_BOX_GATHER.md`` for motivation.  For each k-point, a
lookup table maps every ``(nx, ny, nz)`` FFT-box cell to the
``g``-index within that k's coefficient slab (or a sentinel if the
cell has no coefficient).  At runtime one ``jnp.take`` fills the
whole FFT box — no scatter, no per-k loop.

Public API
----------
``build_g_index_for_fft_box`` — host-side precompute, once per WFN.
``make_fft_box_kernel``       — jitted shard_map kernel, reused per
                                band chunk.
"""
from __future__ import annotations

from typing import Sequence

import jax
import jax.numpy as jnp
import numpy as np
from jax.experimental.shard_map import shard_map
from jax.sharding import Mesh, PartitionSpec as P


__all__ = ["build_g_index_for_fft_box", "make_fft_box_kernel"]


def build_g_index_for_fft_box(
    gvecs_per_k: Sequence[np.ndarray],
    fft_grid: tuple[int, int, int],
    ngkmax: int,
) -> np.ndarray:
    """Precompute ``g_index[k, nx, ny, nz]``.

    For each k and each FFT-box cell ``(nx, ny, nz)``, the entry is the
    position ``g`` in ``[0, ngk[k])`` such that ``gvecs_per_k[k][g] %
    fft_grid == (nx, ny, nz)``, or ``ngkmax`` (a sentinel used by the
    runtime kernel to gather zero for empty cells).

    Parameters
    ----------
    gvecs_per_k : length-``nk`` sequence of ``(ngk[k], 3)`` int arrays
        Integer G-vectors (may be negative; wrapped mod ``fft_grid``).
    fft_grid : (nx, ny, nz)
    ngkmax : int
        Max G-count across all k; also the sentinel value.

    Returns
    -------
    g_index : ``(nk, nx, ny, nz)`` int32
    """
    nk = len(gvecs_per_k)
    nx, ny, nz = (int(v) for v in fft_grid)
    g_index = np.full((nk, nx, ny, nz), ngkmax, dtype=np.int32)

    fft_grid_np = np.asarray(fft_grid, dtype=np.int64)
    for k in range(nk):
        gvecs = np.asarray(gvecs_per_k[k], dtype=np.int32)
        if gvecs.shape[0] > ngkmax:
            raise ValueError(
                f"k={k}: ngk={gvecs.shape[0]} > ngkmax={ngkmax}")
        wrapped = gvecs % fft_grid_np[None, :]
        g_locals = np.arange(gvecs.shape[0], dtype=np.int32)
        g_index[k, wrapped[:, 0], wrapped[:, 1], wrapped[:, 2]] = g_locals
    return g_index


def make_fft_box_kernel(
    mesh: Mesh,
    nk: int,
    ngkmax: int,
    nb_padded: int,
    nspinor: int,
    fft_grid: tuple[int, int, int],
):
    """Build a jitted ``(cnk_slab, g_index) → psi_G_box`` kernel.

    Inputs (global, cnk_slab band-sharded over the combined ``(x,y)``
    mesh axes, g_index replicated):

    ``cnk_slab``     ``(nb_padded, nspinor, nk, ngkmax, 2)`` f64 — the
                     ``re/im``-packed output of ``read_kchunk_union_sharded``
                     with ``kchunk_axis=2``.
    ``g_index``      ``(nk, nx, ny, nz)`` int32 from
                     :func:`build_g_index_for_fft_box`.

    Output:

    ``psi_G_box``    ``(nk, nb_padded, nspinor, nx, ny, nz)`` c128
                     sharded ``P(None, ('x','y'), None, None, None, None)``.
    """
    mesh_x = int(mesh.shape["x"])
    mesh_y = int(mesh.shape["y"])
    world_size = mesh_x * mesh_y
    if nb_padded % world_size != 0:
        raise ValueError(f"nb_padded={nb_padded} not divisible by {world_size}")
    bands_per_rank = nb_padded // world_size
    nx, ny, nz = (int(v) for v in fft_grid)

    def _per_rank(cnk_slab: jax.Array, g_index: jax.Array) -> jax.Array:
        # (bpr, ns, nk, ngkmax, 2) f64 → (bpr, ns, nk, ngkmax) c128
        cnk = cnk_slab[..., 0] + 1j * cnk_slab[..., 1]
        # Append one zero slot so the sentinel index (== ngkmax) gathers
        # zero.  `_padded` has the extra slot on the G axis only.
        zero_slot = jnp.zeros(
            (bands_per_rank, nspinor, nk, 1), dtype=jnp.complex128)
        cnk_padded = jnp.concatenate([cnk, zero_slot], axis=-1)

        # Single flat-axis gather: view cnk_padded as
        # (bpr, ns, nk * (ngkmax + 1)) and look up a k-and-g-combined
        # index per FFT cell.  One kernel launch for every (k, box cell).
        cnk_flat = cnk_padded.reshape(
            bands_per_rank, nspinor, nk * (ngkmax + 1))
        k_stride = ngkmax + 1
        flat_index = (
            jnp.arange(nk, dtype=jnp.int32)[:, None, None, None] * k_stride
            + g_index
        )  # (nk, nx, ny, nz)
        gathered = jnp.take(cnk_flat, flat_index, axis=2)
        # Move nk to the front for the canonical output layout.
        return jnp.transpose(gathered, (2, 0, 1, 3, 4, 5))

    return jax.jit(shard_map(
        _per_rank, mesh=mesh,
        in_specs=(
            P(("x", "y"), None, None, None, None),   # cnk_slab
            P(None, None, None, None),                # g_index
        ),
        out_specs=P(None, ("x", "y"), None, None, None, None),
        check_rep=False,
    ))
