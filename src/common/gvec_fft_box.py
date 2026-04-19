"""Sparse-G → dense-FFT-box via precomputed inverse-index gather.

See ``GVEC_FFT_BOX_GATHER.md`` for the motivation.  Short version: the
straightforward ``psi.at[..., gx, gy, gz].set(cnk)`` scatter is 10–100×
slower on A100 than the equivalent gather.  Flip it: precompute an
``inv[..., nx, ny, nz]`` index map once (host-side, cheap), and at
runtime read from a zero-padded ``cnk`` via one ``jnp.take``.

Public API
----------
:func:`build_within_k_inv_map`
    Build ``inv[k, nx, ny, nz] = g_local`` where each k-point's
    coefficient slab is a contiguous ``[0, ngk[k])`` range; the
    returned map has an ``ngkmax`` sentinel for empty FFT cells.
    Suitable for any caller that holds per-k ngkmax-wide slabs (the
    ``PhdfReadKchunkUnionFfi`` output layout is the motivating example).

:func:`make_gather_fft_box_kernel`
    Build a jitted ``shard_map`` kernel that consumes an
    ``(*leading, nk, ngkmax, 2)`` real+imag slab and an ``inv`` map,
    and produces an ``(*leading, nk, nx, ny, nz)`` complex FFT box.
    Band axis is sharded over ``('x','y')``; everything else
    replicated.
"""
from __future__ import annotations

from typing import Sequence

import jax
import jax.numpy as jnp
import numpy as np
from jax.experimental.shard_map import shard_map
from jax.sharding import Mesh, PartitionSpec as P


__all__ = [
    "build_within_k_inv_map",
    "make_gather_fft_box_kernel",
]


def build_within_k_inv_map(
    gvecs_per_k: Sequence[np.ndarray],
    fft_grid: tuple[int, int, int],
    ngkmax: int,
) -> np.ndarray:
    """Precompute ``inv[k, nx, ny, nz]`` mapping FFT cells to within-slab
    G-indices (with an ``ngkmax`` sentinel for unused cells).

    Parameters
    ----------
    gvecs_per_k : sequence of length ``nk``
        Each element is an ``(ngk[k], 3)`` int array giving this
        k-point's G-vectors in integer reciprocal-lattice coordinates.
        Unwrapped (may be negative) — they get modulo'd by ``fft_grid``
        inside.
    fft_grid : (nx, ny, nz)
        FFT-box dimensions.
    ngkmax : int
        Maximum G-count per k (compile-time constant downstream).
        Must satisfy ``ngkmax >= max(len(gvecs_per_k[k]))``.

    Returns
    -------
    inv : (nk, nx, ny, nz) int32
        ``inv[k, a, b, c] = g_local`` for the unique g in ``[0, ngk[k])``
        such that ``gvecs_per_k[k][g] % fft_grid == (a, b, c)``, or
        ``ngkmax`` (sentinel) if no such g exists.
    """
    nk = len(gvecs_per_k)
    nx, ny, nz = (int(v) for v in fft_grid)
    inv = np.full((nk, nx, ny, nz), ngkmax, dtype=np.int32)

    fft_np = np.asarray(fft_grid, dtype=np.int64)
    for k in range(nk):
        gv = np.asarray(gvecs_per_k[k], dtype=np.int32)
        if gv.shape[0] > ngkmax:
            raise ValueError(
                f"build_within_k_inv_map: k={k} has ngk={gv.shape[0]} > "
                f"ngkmax={ngkmax}")
        wrapped = gv % fft_np[None, :]
        for g_local, (gx, gy, gz) in enumerate(wrapped):
            inv[k, int(gx), int(gy), int(gz)] = int(g_local)
    return inv


def make_gather_fft_box_kernel(
    mesh: Mesh,
    nk: int,
    ngkmax: int,
    nb_padded: int,
    nspinor: int,
    fft_grid: tuple[int, int, int],
):
    """Build a jitted kernel: ``(real+imag slab, inv) → complex FFT box``.

    The kernel does ONE ``jnp.take`` per rank (fused into a single
    scatter-free CUDA kernel by XLA) — no Python unrolling, no per-k
    vmap.  Output is the shard of the dense FFT box for this rank's
    band stripe.

    Input shardings
    ---------------
    slab_real_imag : global ``(nb_padded, nspinor, nk, ngkmax, 2)`` f64.
        Sharded ``P(('x','y'), None, None, None, None)`` — band axis is
        split over the combined ``('x','y')`` mesh axes; nk sits
        between spinor and G so HDF5 row-major iteration of a union
        read visits ``(b, s)`` then all k's in g-order (see
        ``ffi/phdf5/cpp/read_ffi.cc`` for the iteration-order design
        note).
    inv : global ``(nk, nx, ny, nz)`` int32, fully replicated.

    Output sharding
    ---------------
    psi_G : global ``(nk, nb_padded, nspinor, nx, ny, nz)`` c128,
        sharded ``P(None, ('x','y'), None, None, None, None)``.
    """
    p = int(mesh.shape["x"])
    q = int(mesh.shape["y"])
    world = p * q
    if nb_padded % world != 0:
        raise ValueError(
            f"nb_padded={nb_padded} not divisible by world={world}")
    bands_per_rank = nb_padded // world
    nx, ny, nz = (int(v) for v in fft_grid)

    trace_counter = [0]

    def _per_rank(slab_real_imag: jax.Array, inv: jax.Array) -> jax.Array:
        trace_counter[0] += 1
        # slab_real_imag: (bands_per_rank, nspinor, nk, ngkmax, 2) f64
        cnk = slab_real_imag[..., 0] + 1j * slab_real_imag[..., 1]

        # Zero-pad a single extra G-slot at index ngkmax; the inv
        # sentinel (== ngkmax) indexes into it, so empty FFT cells
        # gather exact zero.
        zero_slot = jnp.zeros(
            (bands_per_rank, nspinor, nk, 1), dtype=jnp.complex128)
        cnk_padded = jnp.concatenate([cnk, zero_slot], axis=-1)

        # Flatten the (nk, ngkmax+1) trailing dims so we can express
        # the k-dependent gather as a single 1-D lookup.
        cnk_flat = cnk_padded.reshape(
            bands_per_rank, nspinor, nk * (ngkmax + 1))
        flat_idx = (
            jnp.arange(nk, dtype=jnp.int32)[:, None, None, None]
            * (ngkmax + 1)
            + inv
        )  # (nk, nx, ny, nz)

        gathered = jnp.take(cnk_flat, flat_idx, axis=2)  # (bpr, ns, nk, nx, ny, nz)
        # Move nk to the front for the conventional output layout.
        return jnp.transpose(gathered, (2, 0, 1, 3, 4, 5))

    input_spec_slab = P(("x", "y"), None, None, None, None)
    input_spec_inv = P(None, None, None, None)
    output_spec = P(None, ("x", "y"), None, None, None, None)

    sharded = shard_map(
        _per_rank, mesh=mesh,
        in_specs=(input_spec_slab, input_spec_inv),
        out_specs=output_spec,
        check_rep=False,
    )
    jitted = jax.jit(sharded)
    jitted._trace_counter = trace_counter  # type: ignore[attr-defined]
    return jitted
