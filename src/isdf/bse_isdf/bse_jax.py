"""Sharded ISDF-BSE matrix-vector product for JAX.

This module implements a high-performance, distributed Bethe-Salpeter Equation (BSE)
matrix-vector product using Interpolative Separable Density Fitting (ISDF).

The BSE Hamiltonian in the Tamm-Dancoff Approximation (TDA) is:
    H_BSE = D + 2V - W

where:
    D: Diagonal term from single-particle energy differences (ε_c - ε_v)
    V: Direct (Coulomb) term at q=0
    W: Screened exchange term (k-k' dependent)

Key features:
- Trial vectors X(b, c, v, k) are sharded along conduction bands (c) on the X-axis
- W and V matrices have O(N_μ²/P²) memory per device using 2D (X × Y) mesh
- Spin-traced pair amplitudes: M_cv(μ,k) = Σ_s ψ*_{c,s}(μ,k) ψ_{v,s}(μ,k)
- FFT convolution for k → R → k' momentum transfer

Communication pattern (3 collectives per matvec):
1. psum_X: Complete c-sum in encoding
2. psum_Y: Complete ν-sum after W contraction  
3. reduce_scatter_X: Distribute c in decoding
"""

from __future__ import annotations
import os
from typing import Tuple, Callable, Optional, NamedTuple
from functools import partial
from types import SimpleNamespace
import glob
import math
import sys

import numpy as np
import jax
import jax.numpy as jnp
import h5py
from jax.sharding import Mesh, PartitionSpec as P, NamedSharding
from jax import lax
try:
    from jax import shard_map as _shard_map_fn
except ImportError:  # pragma: no cover - older JAX
    from jax.experimental import shard_map as _shard_map_mod
    _shard_map_fn = _shard_map_mod.shard_map

from isdf.bse_isdf.bse_preconditioner import energy_diff_cv_k

# Enable 64-bit precision
jax.config.update("jax_enable_x64", True)


class BSEData(NamedTuple):
    """Container for BSE calculation data."""
    # Wavefunctions at ISDF centroids: (nk, nb, nspinor, n_rmu)
    psi_c_X: jax.Array  # Conduction, μ on X-axis
    psi_c_Y: jax.Array  # Conduction, μ on Y-axis (as ν)
    psi_v_X: jax.Array  # Valence, μ on X-axis
    psi_v_Y: jax.Array  # Valence, μ on Y-axis (as ν)
    
    # Single-particle energies: (nk, nb)
    eps_c: jax.Array  # Conduction band energies
    eps_v: jax.Array  # Valence band energies
    
    # ISDF matrices
    W_q: jax.Array     # Screened exchange in q-space: (n_rmu, n_rmu, nkx, nky, nkz), P('x', 'y', None, None, None)
    V_q0: jax.Array    # Bare Coulomb at q=0: (n_rmu, n_rmu), P('x', 'y')
    
    # k-grid dimensions
    nkx: int
    nky: int
    nkz: int


def create_mesh_2d(devices: Optional[list] = None) -> Mesh:
    """Create a 2D device mesh for BSE sharding.
    
    Args:
        devices: List of devices. If None, uses all available devices.
        
    Returns:
        Mesh with axes ('x', 'y') for μ/c and ν sharding.
    """
    if devices is None:
        devices = jax.devices()
    
    n_devices = len(devices)
    
    # Find best 2D factorization (prefer square-ish)
    px = int(np.sqrt(n_devices))
    while n_devices % px != 0:
        px -= 1
    py = n_devices // px
    
    device_array = np.array(devices).reshape(px, py)
    return Mesh(device_array, axis_names=('x', 'y'))


def make_bse_shardings(mesh_xy: Mesh) -> SimpleNamespace:
    return SimpleNamespace(
        X=NamedSharding(mesh_xy, P(None, 'x', 'y', None)),
        psi_x=NamedSharding(mesh_xy, P(None, None, None, 'x')),
        psi_y=NamedSharding(mesh_xy, P(None, None, None, 'y')),
        V=NamedSharding(mesh_xy, P('x', 'y')),
        W=NamedSharding(mesh_xy, P('x', 'y', None, None, None)),
        eps=NamedSharding(mesh_xy, P(None, None)),
    )


def build_bse_ring_matvec(mesh_xy: Mesh, nkx: int, nky: int, nkz: int):
    px, py = mesh_xy.devices.shape
    sh = make_bse_shardings(mesh_xy)

    def _matvec(X, psi_c_X, psi_c_Y, psi_v_X, psi_v_Y, eps_c, eps_v, W_q, V_q0):
        return apply_bse_hamiltonian_ring(
            X, nkx, nky, nkz,
            psi_c_X, psi_c_Y, psi_v_X, psi_v_Y,
            eps_c, eps_v, W_q, V_q0,
            px, py,
        )

    return _shard_map_fn(
        _matvec,
        mesh=mesh_xy,
        in_specs=(P(None, 'x', 'y', None), P(None, None, None, 'x'), P(None, None, None, 'y'),
                  P(None, None, None, 'x'), P(None, None, None, 'y'),
                  P(None, None), P(None, None), P('x', 'y', None, None, None), P('x', 'y')),
        out_specs=P(None, 'x', 'y', None),
    )


def _ring_perm(axis_size: int) -> tuple[tuple[int, int], ...]:
    return tuple((i, (i + 1) % axis_size) for i in range(axis_size))


def _pad_to_multiple(x: jax.Array, axis: int, multiple: int) -> tuple[jax.Array, int]:
    size = x.shape[axis]
    pad = (-size) % multiple
    if pad == 0:
        return x, size
    pad_width = [(0, 0)] * x.ndim
    pad_width[axis] = (0, pad)
    return jnp.pad(x, pad_width, mode="constant"), size


def _pad_last_axis(x: jax.Array, target: int) -> jax.Array:
    pad = target - x.shape[-1]
    if pad <= 0:
        return x
    pad_width = [(0, 0)] * x.ndim
    pad_width[-1] = (0, pad)
    return jnp.pad(x, pad_width, mode="constant")


def _pad_last_two_axes(x: jax.Array, target: int) -> jax.Array:
    pad0 = target - x.shape[-2]
    pad1 = target - x.shape[-1]
    if pad0 <= 0 and pad1 <= 0:
        return x
    pad_width = [(0, 0)] * x.ndim
    pad_width[-2] = (0, max(0, pad0))
    pad_width[-1] = (0, max(0, pad1))
    return jnp.pad(x, pad_width, mode="constant")


def _pad_first_two_axes(x: jax.Array, target: int) -> jax.Array:
    pad0 = target - x.shape[0]
    pad1 = target - x.shape[1]
    if pad0 <= 0 and pad1 <= 0:
        return x
    pad_width = [(0, 0)] * x.ndim
    pad_width[0] = (0, max(0, pad0))
    pad_width[1] = (0, max(0, pad1))
    return jnp.pad(x, pad_width, mode="constant")


def _pad_axis_to_multiple(x: jax.Array, axis: int, multiple: int) -> tuple[jax.Array, int]:
    size = x.shape[axis]
    pad = (-size) % multiple
    if pad == 0:
        return x, size
    pad_width = [(0, 0)] * x.ndim
    pad_width[axis] = (0, pad)
    return jnp.pad(x, pad_width, mode="constant"), size


def _ring_sum_valence(
    X: jax.Array,
    psi_v_Y: jax.Array,
    v_chunk: int,
    py: int,
    nu_local: int,
) -> jax.Array:
    """Ring over Y to sum valence contraction into R(b, c_x, k, s, nu_y)."""
    axis_index_y = jnp.asarray(lax.axis_index("y"), dtype=jnp.int32)
    nk, _, nspinor, _ = psi_v_Y.shape

    R0 = jnp.zeros((X.shape[0], X.shape[1], nk, nspinor, nu_local), dtype=X.dtype)
    perm = _ring_perm(py)

    def step(i, carry):
        buf, R = carry
        origin = (axis_index_y - jnp.asarray(i, dtype=jnp.int32)) % py
        v_start = origin * jnp.asarray(v_chunk, dtype=jnp.int32)
        z = jnp.int32(0)
        psi_v_slice = lax.dynamic_slice(
            psi_v_Y, (z, v_start, z, z), (nk, v_chunk, nspinor, nu_local)
        )
        R = R + jnp.einsum("kv sN,bcvk->bcksN", jnp.conj(psi_v_slice), buf)
        buf = lax.ppermute(buf, axis_name="y", perm=perm)
        return buf, R

    R0 = lax.pcast(R0, axis_name=("x", "y"), to="varying")
    _, R_total = lax.fori_loop(0, py, step, (X, R0))
    return R_total


def _ring_sum_conduction(
    R: jax.Array,
    psi_c_X: jax.Array,
    c_chunk: int,
    px: int,
    mu_local: int,
) -> jax.Array:
    """Ring over X to sum conduction contraction into T(b, mu_x, nu_y, t, s, k)."""
    axis_index_x = jnp.asarray(lax.axis_index("x"), dtype=jnp.int32)
    nk, _, nspinor, _ = psi_c_X.shape
    nu_local = R.shape[4]

    T0 = jnp.zeros((R.shape[0], mu_local, nu_local, nspinor, nspinor, nk), dtype=R.dtype)
    perm = _ring_perm(px)

    def step(i, carry):
        buf, T = carry
        origin = (axis_index_x - jnp.asarray(i, dtype=jnp.int32)) % px
        c_start = origin * jnp.asarray(c_chunk, dtype=jnp.int32)
        z = jnp.int32(0)
        psi_c_slice = lax.dynamic_slice(
            psi_c_X, (z, c_start, z, z), (nk, c_chunk, nspinor, mu_local)
        )
        T = T + jnp.einsum("kctM,bcksN->bMNtsk", psi_c_slice, buf)
        buf = lax.ppermute(buf, axis_name="x", perm=perm)
        return buf, T

    T0 = lax.pcast(T0, axis_name=("x", "y"), to="varying")
    _, T_total = lax.fori_loop(0, px, step, (R, T0))
    return T_total


def apply_W_ring(
    X: jax.Array,
    psi_c_X: jax.Array,
    psi_v_Y: jax.Array,
    W_q: jax.Array,
    nkx: int,
    nky: int,
    nkz: int,
    px: int,
    py: int,
) -> jax.Array:
    """Apply screened exchange using ring communication over X/Y."""
    nk = nkx * nky * nkz
    nb_trial, nc_local, nv_local, _ = X.shape
    sqrt_nk = jnp.sqrt(jnp.asarray(nk, dtype=jnp.float64))

    c_chunk = X.shape[1]
    v_chunk = X.shape[2]

    # ----- Encode via ring reductions -----
    n_rmu_local_X = psi_c_X.shape[-1]
    n_rmu_local_Y = psi_v_Y.shape[-1]
    R = _ring_sum_valence(X, psi_v_Y, v_chunk, py, n_rmu_local_Y)
    T = _ring_sum_conduction(R, psi_c_X, c_chunk, px, n_rmu_local_X)

    # ----- Convolution in k using FFT -----
    nspinor = psi_c_X.shape[2]

    T_k = T.reshape(nb_trial, n_rmu_local_X, n_rmu_local_Y, nspinor, nspinor, nkx, nky, nkz)
    T_R = jnp.fft.ifftn(T_k, axes=(5, 6, 7), norm="ortho")
    W_R = jnp.fft.ifftn(W_q, axes=(2, 3, 4), norm="ortho")
    U_R = W_R[None, :, :, None, None, :, :, :] * T_R
    U_q = jnp.fft.fftn(U_R, axes=(5, 6, 7), norm="ortho")
    U = U_q.reshape(nb_trial, n_rmu_local_X, n_rmu_local_Y, nspinor, nspinor, nk)

    # ----- Decode: reduce-scatter over X then Y -----
    A_partial = jnp.einsum("kctM,bMNtsk->bcNsk", jnp.conj(psi_c_X), U)
    D = lax.psum_scatter(A_partial, axis_name="x", scatter_dimension=1, tiled=True)
    WX_partial = jnp.einsum("kvsN,bcNsk->bcvk", psi_v_Y, D)
    WX = lax.psum_scatter(WX_partial, axis_name="y", scatter_dimension=2, tiled=True)

    return WX / sqrt_nk


def apply_V_ring(
    X: jax.Array,
    psi_c_Y: jax.Array,
    psi_v_Y: jax.Array,
    psi_c_X: jax.Array,
    psi_v_X: jax.Array,
    V_q0: jax.Array,
    nk: int,
    px: int,
    py: int,
) -> jax.Array:
    """Apply direct term using ring communication over Y (v) then X (c)."""
    nb_trial, _, _, _ = X.shape
    sqrt_nk = jnp.sqrt(jnp.asarray(nk, dtype=jnp.float64))

    c_chunk = X.shape[1]
    v_chunk = X.shape[2]

    axis_index_x = jnp.asarray(lax.axis_index("x"), dtype=jnp.int32)
    axis_index_y = jnp.asarray(lax.axis_index("y"), dtype=jnp.int32)

    nk_local = psi_c_Y.shape[0]
    nspinor = psi_c_Y.shape[2]
    nu_local = psi_c_Y.shape[-1]
    mu_local = psi_c_X.shape[-1]

    # ----- Ring over Y: sum over v for local c chunk -----
    c_start_local = axis_index_x * jnp.asarray(c_chunk, dtype=jnp.int32)
    z = jnp.int32(0)
    psi_c_slice = lax.dynamic_slice(
        psi_c_Y, (z, c_start_local, z, z), (nk_local, c_chunk, nspinor, nu_local)
    )

    A0 = jnp.zeros((nb_trial, c_chunk, nu_local, nk_local), dtype=X.dtype)
    perm_y = _ring_perm(py)

    def step_y(i, carry):
        buf, A = carry
        origin = (axis_index_y - jnp.asarray(i, dtype=jnp.int32)) % py
        v_start = origin * jnp.asarray(v_chunk, dtype=jnp.int32)
        psi_v_slice = lax.dynamic_slice(
            psi_v_Y, (z, v_start, z, z), (nk_local, v_chunk, nspinor, nu_local)
        )
        R_v = jnp.einsum("kvsN,bcvk->bcksN", psi_v_slice, buf)
        A = A + jnp.einsum("kcsN,bcksN->bcNk", jnp.conj(psi_c_slice), R_v)
        buf = lax.ppermute(buf, axis_name="y", perm=perm_y)
        return buf, A

    A0 = lax.pcast(A0, axis_name=("x", "y"), to="varying")
    _, A_local = lax.fori_loop(0, py, step_y, (X, A0))

    # ----- Ring over X: sum over c to build S(b, nu, k) -----
    S0 = jnp.zeros((nb_trial, nu_local, nk_local), dtype=X.dtype)
    perm_x = _ring_perm(px)

    def step_x(i, carry):
        buf, S = carry
        S = S + jnp.sum(buf, axis=1)
        buf = lax.ppermute(buf, axis_name="x", perm=perm_x)
        return buf, S

    S0 = lax.pcast(S0, axis_name=("x", "y"), to="varying")
    _, S_total = lax.fori_loop(0, px, step_x, (A_local, S0))

    # Split 1/Nk across encode/decode
    S_total = S_total / sqrt_nk

    # ----- Apply V(μ,ν) at q=0 -----
    U_partial = jnp.einsum("MN,bNk->bMk", V_q0, S_total)
    U = lax.psum(U_partial, axis_name="y")

    # ----- Decode: local v chunk, reduce-scatter over X for c -----
    v_start_local = axis_index_y * jnp.asarray(v_chunk, dtype=jnp.int32)
    psi_v_slice_X = lax.dynamic_slice(
        psi_v_X, (z, v_start_local, z, z), (nk_local, v_chunk, nspinor, mu_local)
    )
    M_X = jnp.einsum("kcsm,kvsm->kcvm", jnp.conj(psi_c_X), psi_v_slice_X)
    VX_partial = jnp.einsum("kcvM,bMk->bcvk", jnp.conj(M_X), U)
    VX = lax.psum_scatter(VX_partial, axis_name="x", scatter_dimension=1, tiled=True)

    return VX / sqrt_nk


def apply_bse_hamiltonian_ring(
    X: jax.Array,
    nkx: int,
    nky: int,
    nkz: int,
    psi_c_X: jax.Array,
    psi_c_Y: jax.Array,
    psi_v_X: jax.Array,
    psi_v_Y: jax.Array,
    eps_c: jax.Array,
    eps_v: jax.Array,
    W_q: jax.Array,
    V_q0: jax.Array,
    px: int,
    py: int,
) -> jax.Array:
    nk = nkx * nky * nkz
    if X.shape[1] % px != 0 or X.shape[2] % py != 0:
        raise ValueError("X band dimensions must be divisible by px/py for ring matvec")

    eps_c_pad = eps_c
    eps_v_pad = eps_v
    X_pad = X
    axis_index_x = jnp.asarray(lax.axis_index("x"), dtype=jnp.int32)
    axis_index_y = jnp.asarray(lax.axis_index("y"), dtype=jnp.int32)
    c_chunk = X_pad.shape[1]
    v_chunk = X_pad.shape[2]
    c_chunk_i = jnp.asarray(c_chunk, dtype=jnp.int32)
    v_chunk_i = jnp.asarray(v_chunk, dtype=jnp.int32)
    z = jnp.int32(0)
    eps_c_local = lax.dynamic_slice(eps_c_pad, (z, axis_index_x * c_chunk_i), (nk, c_chunk))
    eps_v_local = lax.dynamic_slice(eps_v_pad, (z, axis_index_y * v_chunk_i), (nk, v_chunk))
    delta_E = eps_c_local.T[None, :, None, :] - eps_v_local.T[None, None, :, :]
    D_term = delta_E * X_pad
    V_term = apply_V_ring(X_pad, psi_c_Y, psi_v_Y, psi_c_X, psi_v_X, V_q0, nk, px, py)
    W_term = apply_W_ring(X_pad, psi_c_X, psi_v_Y, W_q, nkx, nky, nkz, px, py)

    HX = D_term + V_term - W_term
    return HX


def _get_local_mesh_coords(mesh_xy: Mesh) -> tuple[list[tuple[int, int]], int, int]:
    devices_2d = np.asarray(mesh_xy.devices)
    grid_x, grid_y = devices_2d.shape
    local_devices = list(jax.local_devices())
    local_coords = [tuple(np.argwhere(devices_2d == d)[0]) for d in local_devices]
    local_coords = sorted(local_coords, key=lambda c: c[0] * grid_y + c[1])
    return local_coords, grid_x, grid_y


def _get_local_axis_coords(local_coords: list[tuple[int, int]]) -> tuple[list[int], list[int]]:
    local_x = sorted({coord[0] for coord in local_coords})
    local_y = sorted({coord[1] for coord in local_coords})
    return local_x, local_y


def _assert_local_block(local_coords: list[tuple[int, int]], local_x: list[int], local_y: list[int]) -> None:
    expected = {(x, y) for x in local_x for y in local_y}
    actual = set(local_coords)
    if actual != expected:
        raise ValueError(
            "Local devices are not a full x/y block; shard-aware loader expects "
            "local device coords to form a Cartesian product."
        )


def _read_psi_mu_sharded(
    dset: h5py.Dataset,
    band_indices: np.ndarray,
    mu_per_shard: int,
    axis: str,
    mesh_xy: Mesh,
    n_rmu_pad: int,
    dtype: np.dtype = np.complex128,
    trim: bool = True,
) -> jax.Array:
    local_coords, grid_x, grid_y = _get_local_mesh_coords(mesh_xy)
    local_x, local_y = _get_local_axis_coords(local_coords)
    _assert_local_block(local_coords, local_x, local_y)
    n_rmu = dset.shape[-1]

    if axis == "x":
        local_axis_coords = local_x
        n_axis = grid_x
    elif axis == "y":
        local_axis_coords = local_y
        n_axis = grid_y
    else:
        raise ValueError("axis must be 'x' or 'y'")

    nk, _, nspinor, _ = dset.shape
    nb = len(band_indices)
    local_mu = mu_per_shard * len(local_axis_coords)
    local_psi = np.zeros((nk, nb, nspinor, local_mu), dtype=dtype)

    for i, coord in enumerate(local_axis_coords):
        mu_start = coord * mu_per_shard
        mu_end = min(mu_start + mu_per_shard, n_rmu)
        if mu_start >= n_rmu:
            continue
        slab = dset[:, band_indices, :, mu_start:mu_end]
        if slab.shape[-1] < mu_per_shard:
            pad = mu_per_shard - slab.shape[-1]
            slab = np.pad(slab, ((0, 0), (0, 0), (0, 0), (0, pad)), mode="constant")
        mu_off = i * mu_per_shard
        local_psi[:, :, :, mu_off:mu_off + mu_per_shard] = slab

    global_shape = (nk, nb, nspinor, n_rmu_pad)
    psi_sharding = NamedSharding(mesh_xy, P(None, None, None, axis))
    local_psi_jax = jax.device_put(local_psi)
    psi_global = jax.make_array_from_process_local_data(psi_sharding, local_psi_jax, global_shape)
    if trim and global_shape[-1] > n_rmu:
        psi_global = psi_global[..., :n_rmu]
    return psi_global


def _read_vq0_sharded(
    dset: h5py.Dataset,
    mu_per_x: int,
    nu_per_y: int,
    mesh_xy: Mesh,
    n_rmu_pad: int,
    dtype: np.dtype = np.complex128,
    trim: bool = True,
) -> jax.Array:
    local_coords, grid_x, grid_y = _get_local_mesh_coords(mesh_xy)
    local_x, local_y = _get_local_axis_coords(local_coords)
    _assert_local_block(local_coords, local_x, local_y)
    n_rmu = dset.shape[6]
    n_rnu = dset.shape[7]

    local_mu = mu_per_x * len(local_x)
    local_nu = nu_per_y * len(local_y)
    local_v = np.zeros((local_mu, local_nu), dtype=dtype)

    for ix, x_coord in enumerate(local_x):
        mu_start = x_coord * mu_per_x
        mu_end = min(mu_start + mu_per_x, n_rmu)
        if mu_start >= n_rmu:
            continue
        for iy, y_coord in enumerate(local_y):
            nu_start = y_coord * nu_per_y
            nu_end = min(nu_start + nu_per_y, n_rnu)
            if nu_start >= n_rnu:
                continue
            slab = dset[0, 0, 0, 0, 0, 0, mu_start:mu_end, nu_start:nu_end]
            if slab.shape[0] < mu_per_x or slab.shape[1] < nu_per_y:
                pad_mu = mu_per_x - slab.shape[0]
                pad_nu = nu_per_y - slab.shape[1]
                slab = np.pad(slab, ((0, pad_mu), (0, pad_nu)), mode="constant")
            mu_off = ix * mu_per_x
            nu_off = iy * nu_per_y
            local_v[mu_off:mu_off + mu_per_x, nu_off:nu_off + nu_per_y] = slab

    global_shape = (n_rmu_pad, n_rmu_pad)
    v_sharding = NamedSharding(mesh_xy, P("x", "y"))
    local_v_jax = jax.device_put(local_v)
    v_global = jax.make_array_from_process_local_data(v_sharding, local_v_jax, global_shape)
    if trim and (global_shape[0] > n_rmu or global_shape[1] > n_rnu):
        v_global = v_global[:n_rmu, :n_rnu]
    return v_global


def _read_wq_sharded(
    dset: h5py.Dataset,
    mu_per_x: int,
    nu_per_y: int,
    mesh_xy: Mesh,
    n_rmu_pad: int,
    dtype: np.dtype = np.complex128,
    trim: bool = True,
) -> jax.Array:
    local_coords, grid_x, grid_y = _get_local_mesh_coords(mesh_xy)
    local_x, local_y = _get_local_axis_coords(local_coords)
    _assert_local_block(local_coords, local_x, local_y)
    n_rmu = dset.shape[6]
    n_rnu = dset.shape[7]
    nkx, nky, nkz = dset.shape[3:6]

    local_mu = mu_per_x * len(local_x)
    local_nu = nu_per_y * len(local_y)
    local_w = np.zeros((local_mu, local_nu, nkx, nky, nkz), dtype=dtype)

    for ix, x_coord in enumerate(local_x):
        mu_start = x_coord * mu_per_x
        mu_end = min(mu_start + mu_per_x, n_rmu)
        if mu_start >= n_rmu:
            continue
        for iy, y_coord in enumerate(local_y):
            nu_start = y_coord * nu_per_y
            nu_end = min(nu_start + nu_per_y, n_rnu)
            if nu_start >= n_rnu:
                continue
            slab = dset[0, 0, 0, :, :, :, mu_start:mu_end, nu_start:nu_end]
            slab = np.transpose(slab, (3, 4, 0, 1, 2))
            if slab.shape[0] < mu_per_x or slab.shape[1] < nu_per_y:
                pad_mu = mu_per_x - slab.shape[0]
                pad_nu = nu_per_y - slab.shape[1]
                slab = np.pad(slab, ((0, pad_mu), (0, pad_nu), (0, 0), (0, 0), (0, 0)), mode="constant")
            mu_off = ix * mu_per_x
            nu_off = iy * nu_per_y
            local_w[mu_off:mu_off + mu_per_x, nu_off:nu_off + nu_per_y, :, :, :] = slab

    global_shape = (n_rmu_pad, n_rmu_pad, nkx, nky, nkz)
    w_sharding = NamedSharding(mesh_xy, P("x", "y", None, None, None))
    local_w_jax = jax.device_put(local_w)
    w_global = jax.make_array_from_process_local_data(w_sharding, local_w_jax, global_shape)
    if trim and (global_shape[0] > n_rmu or global_shape[1] > n_rnu):
        w_global = w_global[:n_rmu, :n_rnu, :, :, :]
    return w_global


def load_bse_data_from_restart_sharded(
    restart_file: str,
    n_val: int = 4,
    n_cond: int = 4,
    fermi_energy: float = 0.0,
    mesh_xy: Optional[Mesh] = None,
    pad_bands: bool = True,
) -> dict:
    """Load BSE inputs from a COHSEX restart file using sharded HDF5 reads.

    This loader only materializes the local shard per process and returns
    X- and Y-sharded wavefunctions along the μ-axis.
    """
    if mesh_xy is None:
        mesh_xy = create_mesh_2d()

    with h5py.File(restart_file, "r") as f:
        vq_dset = f["V_qmunu"]
        if "W0_qmunu" in f and bool(f["W0_qmunu"].attrs.get("W0_ready", False)):
            wq_dset = f["W0_qmunu"]
        else:
            wq_dset = vq_dset
        psi_l_dset = f["psi_l"]
        psi_r_dset = f["psi_r"]
        enk_l = np.asarray(f["enk_l"][:])
        enk_r = np.asarray(f["enk_r"][:])

        nkx, nky, nkz = vq_dset.shape[3:6]
        n_rmu = int(vq_dset.shape[6])
        n_rnu = int(vq_dset.shape[7])
        if n_rmu != n_rnu:
            raise ValueError("Expected square μ/ν dimensions in V_qmunu")

        mean_enk_l = np.mean(enk_l, axis=0)
        mean_enk_r = np.mean(enk_r, axis=0)
        val_mask_l = mean_enk_l < fermi_energy
        cond_mask_r = mean_enk_r > fermi_energy
        n_val_available = int(np.sum(val_mask_l))
        n_cond_available = int(np.sum(cond_mask_r))
        n_val = min(n_val, n_val_available)
        n_cond = min(n_cond, n_cond_available)
        if n_val == 0 or n_cond == 0:
            raise ValueError("No valence or conduction bands found for given Fermi energy")
        val_indices = np.argsort(np.where(val_mask_l, mean_enk_l, -np.inf))[-n_val:]
        cond_indices = np.argsort(np.where(cond_mask_r, mean_enk_r, np.inf))[:n_cond]

        eps_v = jnp.asarray(enk_l[:, val_indices])
        eps_c = jnp.asarray(enk_r[:, cond_indices])

        _, grid_x, grid_y = _get_local_mesh_coords(mesh_xy)
        lcm_xy = math.lcm(grid_x, grid_y)
        n_rmu_pad = ((n_rmu + lcm_xy - 1) // lcm_xy) * lcm_xy
        mu_per_x = n_rmu_pad // grid_x
        nu_per_y = n_rmu_pad // grid_y

        psi_v_X = _read_psi_mu_sharded(psi_l_dset, val_indices, mu_per_x, "x", mesh_xy, n_rmu_pad, trim=False)
        psi_c_X = _read_psi_mu_sharded(psi_r_dset, cond_indices, mu_per_x, "x", mesh_xy, n_rmu_pad, trim=False)

        if pad_bands:
            psi_v_X, n_val_pad = _pad_axis_to_multiple(psi_v_X, axis=1, multiple=grid_y)
            psi_c_X, n_cond_pad = _pad_axis_to_multiple(psi_c_X, axis=1, multiple=grid_x)
            eps_v, _ = _pad_axis_to_multiple(eps_v, axis=1, multiple=grid_y)
            eps_c, _ = _pad_axis_to_multiple(eps_c, axis=1, multiple=grid_x)
        else:
            n_val_pad = int(psi_v_X.shape[1])
            n_cond_pad = int(psi_c_X.shape[1])
        psi_v_Y = jax.lax.with_sharding_constraint(psi_v_X, NamedSharding(mesh_xy, P(None, None, None, "y")))
        psi_c_Y = jax.lax.with_sharding_constraint(psi_c_X, NamedSharding(mesh_xy, P(None, None, None, "y")))

        V_q0 = _read_vq0_sharded(vq_dset, mu_per_x, nu_per_y, mesh_xy, n_rmu_pad, trim=False)
        W_q = _read_wq_sharded(wq_dset, mu_per_x, nu_per_y, mesh_xy, n_rmu_pad, trim=False)

    return {
        "psi_c_X": psi_c_X,
        "psi_c_Y": psi_c_Y,
        "psi_v_X": psi_v_X,
        "psi_v_Y": psi_v_Y,
        "eps_c": eps_c,
        "eps_v": eps_v,
        "W_q": W_q,
        "V_q0": V_q0,
        "nkx": nkx,
        "nky": nky,
        "nkz": nkz,
        "n_rmu": n_rmu,
        "n_rmu_pad": n_rmu_pad,
        "n_val": n_val,
        "n_cond": n_cond,
        "n_val_pad": n_val_pad,
        "n_cond_pad": n_cond_pad,
        "fermi_energy": fermi_energy,
    }


def symmetrize_W_q(
    W_q: jax.Array,
    nkx: int,
    nky: int,
    nkz: int,
) -> jax.Array:
    """Symmetrize W(q) to enforce W(q) = W(-q)† for Hermitian BSE.
    
    For the BSE Hamiltonian to be Hermitian, the screened interaction
    must satisfy time-reversal symmetry: W(q)[μ,ν] = W(-q)[ν,μ]*
    
    This function computes: W_sym(q) = (W(q) + W(-q)†) / 2
    
    Args:
        W_q: (n_rmu, n_rmu, nkx, nky, nkz) interaction in q-space
        nkx, nky, nkz: k-grid dimensions
        
    Returns:
        W_q_sym: Symmetrized interaction
    """
    n_rmu = W_q.shape[0]
    
    # Create index arrays for -q: -q = (nkx-qx, nky-qy, nkz-qz) mod nk
    # Note: q=0 maps to itself
    qx = jnp.arange(nkx)
    qy = jnp.arange(nky)
    qz = jnp.arange(nkz)
    
    minus_qx = (nkx - qx) % nkx
    minus_qy = (nky - qy) % nky
    minus_qz = (nkz - qz) % nkz
    
    # Get W(-q) by advanced indexing
    # W_q shape: (n_rmu, n_rmu, nkx, nky, nkz)
    # We need W_q[:, :, minus_qx, minus_qy, minus_qz] for all combinations
    W_minus_q = W_q[:, :, minus_qx[:, None, None], minus_qy[None, :, None], minus_qz[None, None, :]]
    
    # W(-q)† means conjugate transpose in (μ,ν): swap axes 0,1 and conjugate
    W_minus_q_dag = jnp.conj(W_minus_q).transpose(1, 0, 2, 3, 4)
    
    # Symmetrize
    W_q_sym = (W_q + W_minus_q_dag) / 2
    
    return W_q_sym


def compute_pair_amplitude(psi_c: jax.Array, psi_v: jax.Array) -> jax.Array:
    """Compute spin-traced pair amplitude M_cv(μ,k).
    
    M_cv(k, μ) = Σ_s ψ*_{c,s}(μ,k) ψ_{v,s}(μ,k)
    
    Args:
        psi_c: (nk, nc, nspinor, n_rmu) conduction wavefunctions
        psi_v: (nk, nv, nspinor, n_rmu) valence wavefunctions
        
    Returns:
        M: (nk, nc, nv, n_rmu) spin-traced pair amplitudes
    """
    # Contract over spin: M(k,c,v,μ) = Σ_s conj(ψ_c[k,c,s,μ]) * ψ_v[k,v,s,μ]
    return jnp.einsum('kcsm,kvsm->kcvm', jnp.conj(psi_c), psi_v)


@partial(jax.jit, static_argnums=(1, 2, 3))
def apply_bse_hamiltonian(
    X: jax.Array,
    nkx: int,
    nky: int, 
    nkz: int,
    psi_c_X: jax.Array,
    psi_c_Y: jax.Array,
    psi_v_X: jax.Array,
    psi_v_Y: jax.Array,
    eps_c: jax.Array,
    eps_v: jax.Array,
    W_q: jax.Array,
    V_q0: jax.Array,
) -> jax.Array:
    """Apply BSE Hamiltonian to trial vectors.
    
    This is the main computational kernel for iterative BSE eigensolvers.
    Handles batched trial vectors with X sharded on conduction bands.
    
    Args:
        X: (nb_trial, nc, nv, nk) trial vectors, nc sharded on 'x' axis
        nkx, nky, nkz: k-grid dimensions
        psi_c_X: (nk, nc, nspinor, n_rmu) conduction wfns, n_rmu on 'x'
        psi_c_Y: (nk, nc, nspinor, n_rmu) conduction wfns, n_rmu on 'y' 
        psi_v_X: (nk, nv, nspinor, n_rmu) valence wfns, n_rmu on 'x'
        psi_v_Y: (nk, nv, nspinor, n_rmu) valence wfns, n_rmu on 'y'
        eps_c: (nk, nc) conduction energies
        eps_v: (nk, nv) valence energies
        W_q: (n_rmu, n_rmu, nkx, nky, nkz) screened exchange in q-space
        V_q0: (n_rmu, n_rmu) bare Coulomb at q=0
        
    Returns:
        HX: (nb_trial, nc, nv, nk) result of H @ X, same sharding as X
    """
    nk = nkx * nky * nkz
    nb_trial = X.shape[0]
    
    # Apply each term
    D_term = apply_D(X, eps_c, eps_v)
    V_term = apply_V(X, psi_c_X, psi_c_Y, psi_v_X, psi_v_Y, V_q0, nk)
    W_term = apply_W(X, psi_c_X, psi_c_Y, psi_v_X, psi_v_Y, W_q, nkx, nky, nkz)
    
    # Spinor default: H = D + V - W.
    #
    # IMPORTANT: V and W are applied with different internal contractions for spinors.
    # - V (direct) uses the usual spin-traced cv pair density at each vertex.
    # - W (exchange-like screened direct kernel in Henneke eq (2-16)/(4-6)) is applied via
    #   an intermediate 2x2 spin matrix built from a conduction spinor at μ and a valence
    #   spinor at ν, then contracted with the external (c,v) spinors after applying W.
    return D_term + V_term - W_term


def apply_D(
    X: jax.Array,
    eps_c: jax.Array, 
    eps_v: jax.Array,
) -> jax.Array:
    """Apply diagonal term: [DX](b,c,v,k) = (ε_c(k) - ε_v(k)) X(b,c,v,k).
    
    This is purely local - no communication needed.
    
    Args:
        X: (nb_trial, nc, nv, nk) trial vectors
        eps_c: (nk, nc) conduction energies
        eps_v: (nk, nv) valence energies
        
    Returns:
        DX: (nb_trial, nc, nv, nk)
    """
    # Energy difference: (nc, nv, nk) -> broadcast with X
    delta_E = energy_diff_cv_k(eps_c, eps_v)[None, :, :, :]
    return delta_E * X


def apply_V(
    X: jax.Array,
    psi_c_X: jax.Array,
    psi_c_Y: jax.Array,
    psi_v_X: jax.Array,
    psi_v_Y: jax.Array,
    V_q0: jax.Array,
    nk: int,
) -> jax.Array:
    """Apply direct (Coulomb) term at q=0.
    
    [VX](b,c,v,k) = (1/Nk) Σ_{c'v'} M*_cv(k,μ) V(μ,ν) M_{c'v'}(k,ν) X(b,c',v',k)
    
    Note: V term is local in k (only q=0 contributes to optical response).
    The 1/Nk factor is applied here (not in FFT normalization).
    
    Communication: psum_X (encode) + psum_Y (V contract) + reduce_scatter_X (decode)
    
    Args:
        X: (nb_trial, nc, nv, nk) trial vectors, c on 'x'
        psi_c_X/Y: (nk, nc, nspinor, n_rmu) conduction wfns
        psi_v_X/Y: (nk, nv, nspinor, n_rmu) valence wfns  
        V_q0: (n_rmu, n_rmu) Coulomb at q=0, P('x', 'y')
        nk: total number of k-points
        
    Returns:
        VX: (nb_trial, nc, nv, nk)
    """
    nb_trial, nc_local, nv, nk_flat = X.shape
    
    # 1. Compute pair amplitudes M at ν-points (Y-sharded)
    # M_Y: (nk, nc, nv, n_rmu/Py)
    M_Y = compute_pair_amplitude(psi_c_Y, psi_v_Y)
    
    # 2. Encode: project X onto ν basis
    # S_partial(b, ν_Y, k) = Σ_{c' ∈ local, v'} M(k, c', v', ν_Y) * X(b, c'_X, v', k)
    # X is (b, c_X, v, k), M_Y is (k, c, v, ν_Y)
    # Need to match c indices - M has full c, X has local c
    # Result: (b, ν_Y, k)
    S_partial = jnp.einsum('kcvN,bcvk->bNk', M_Y, X)
    
    # Complete c-sum across X-axis
    S = lax.psum(S_partial, axis_name='x')  # (b, ν_Y, k)

    # Split the overall 1/Nk BZ prefactor symmetrically as 1/sqrt(Nk) on encode and decode.
    sqrt_nk = jnp.sqrt(jnp.asarray(nk, dtype=jnp.float64))
    S = S / sqrt_nk
    
    # 3. Apply V(μ,ν) at q=0
    # U_partial(b, μ_X, k) = Σ_{ν ∈ local} V(μ_X, ν_Y) S(b, ν_Y, k)
    U_partial = jnp.einsum('MN,bNk->bMk', V_q0, S)
    
    # Complete ν-sum across Y-axis  
    U = lax.psum(U_partial, axis_name='y')  # (b, μ_X, k)
    
    # 4. Decode: project back to (c,v) space
    # M_X: (nk, nc, nv, n_rmu/Px)
    M_X = compute_pair_amplitude(psi_c_X, psi_v_X)
    
    # [VX]_full(b, c, v, k) = Σ_{μ ∈ local} conj(M_X)(k, c, v, μ_X) * U(b, μ_X, k)
    VX_partial = jnp.einsum('kcvM,bMk->bcvk', jnp.conj(M_X), U)
    
    # reduce_scatter: sum over μ (X-axis) and scatter c
    # This completes the μ-sum while distributing c across devices
    VX = lax.psum_scatter(VX_partial, axis_name='x', scatter_dimension=1, tiled=True)

    # Second 1/sqrt(Nk) factor (decode side)
    return VX / sqrt_nk


def apply_W(
    X: jax.Array,
    psi_c_X: jax.Array,
    psi_c_Y: jax.Array,
    psi_v_X: jax.Array,
    psi_v_Y: jax.Array,
    W_q: jax.Array,
    nkx: int,
    nky: int,
    nkz: int,
) -> jax.Array:
    """Apply screened exchange term with momentum transfer.
    
    This implements the contraction pattern in Henneke (2020) eq (4-6), i.e.
    the kernel connects a (c,c') bilinear at r=μ and a (v',v) bilinear at r'=ν.
    For spinors, the efficient reordering naturally produces a 2x2 spin matrix:
        T_{t s}(μ,ν,k') = Σ_{c',v'} ψ_{c',t}(μ,k') * ψ*_{v',s}(ν,k') * X(c',v',k')
    (t,s are spinor component indices associated with μ and ν, respectively).
    W is spin-independent (scalar), so it multiplies each spin-matrix element.
    The final decode contracts this spin matrix with the external (c,v) spinors.
    
    Uses FFT convolution with norm='ortho' (unitary FFT).
    With unitary FFTs, the convolution theorem introduces a 1/sqrt(Nk) scaling,
    so we apply an extra 1/sqrt(Nk) prefactor to recover the physical 1/Nk.
    
    Communication: psum_X (encode) + psum_Y (W contract) + reduce_scatter_X (decode)
    
    Args:
        X: (nb_trial, nc, nv, nk) trial vectors, c on 'x'
        psi_c_X/Y: (nk, nc, nspinor, n_rmu) conduction wfns
        psi_v_X/Y: (nk, nv, nspinor, n_rmu) valence wfns
        W_q: (n_rmu, n_rmu, nkx, nky, nkz) screened exchange in q-space, P('x', 'y', ...)
        nkx, nky, nkz: k-grid dimensions
        
    Returns:
        WX: (nb_trial, nc, nv, nk)
    """
    nk = nkx * nky * nkz
    nb_trial, nc_local, nv, _ = X.shape
    sqrt_nk = jnp.sqrt(jnp.asarray(nk, dtype=jnp.float64))
    
    nspinor = psi_c_X.shape[2]
    n_rmu_local_Y = psi_v_Y.shape[-1]
    n_rmu_local_X = psi_c_X.shape[-1]

    # ----- Encode (k-space): build spin-matrix T(b, μ_X, ν_Y, t, s, k) -----
    # R_partial(b, c_local, k, s, ν_Y) = Σ_v conj(ψ_v_Y(k,v,s,ν_Y)) * X(b,c_local,v,k)
    R_partial = jnp.einsum('kv sN,bcvk->bcksN', jnp.conj(psi_v_Y), X)

    # T_partial(b, μ_X, ν_Y, t, s, k) = Σ_{c in local shard} ψ_c_X(k,c,t,μ_X) * R_partial(b,c,k,s,ν_Y)
    T_partial = jnp.einsum('kctM,bcksN->bMNtsk', psi_c_X, R_partial)

    # Complete c-sum across X-axis (c has been eliminated, so this is safe)
    T = lax.psum(T_partial, axis_name='x')  # (b, μ_X, ν_Y, t, s, nk)

    # ----- Convolution in k using FFT (elementwise in μ,ν,t,s) -----
    T_k = T.reshape(nb_trial, n_rmu_local_X, n_rmu_local_Y, nspinor, nspinor, nkx, nky, nkz)
    T_R = jnp.fft.ifftn(T_k, axes=(5, 6, 7), norm='ortho')

    W_R = jnp.fft.ifftn(W_q, axes=(2, 3, 4), norm='ortho')  # (μ_X, ν_Y, nkx, nky, nkz)
    U_R = W_R[None, :, :, None, None, :, :, :] * T_R

    U_q = jnp.fft.fftn(U_R, axes=(5, 6, 7), norm='ortho')
    U = U_q.reshape(nb_trial, n_rmu_local_X, n_rmu_local_Y, nspinor, nspinor, nk)

    # ----- Decode: contract spin matrix with external (c,v) spinors -----
    # A_partial(b, c, ν_Y, s, k) = Σ_{μ_X,t} conj(ψ_c_X(k,c,t,μ_X)) * U(b,μ_X,ν_Y,t,s,k)
    A_partial = jnp.einsum('kctM,bMNtsk->bcNsk', jnp.conj(psi_c_X), U)

    # WX_partial(b, c, v, k) = Σ_{ν_Y,s} ψ_v_Y(k,v,s,ν_Y) * A_partial(b,c,ν_Y,s,k)
    WX_partial = jnp.einsum('kvsN,bcNsk->bcvk', psi_v_Y, A_partial)

    # Complete ν sum across Y-axis (ν has been eliminated by the contraction above)
    WX_nu = lax.psum(WX_partial, axis_name='y')

    # Sum over μ contributions across X-axis and scatter c back onto X sharding
    WX = lax.psum_scatter(WX_nu, axis_name='x', scatter_dimension=1, tiled=True)

    # Apply the remaining 1/sqrt(Nk) to recover the physical 1/Nk prefactor.
    return WX / sqrt_nk


# ============== Single-device version for testing ==============

def apply_bse_hamiltonian_single_device(
    X: jax.Array,
    psi_c: jax.Array,
    psi_v: jax.Array,
    eps_c: jax.Array,
    eps_v: jax.Array,
    W_q: jax.Array,
    V_q0: jax.Array,
    nkx: int,
    nky: int,
    nkz: int,
) -> jax.Array:
    """Single-device BSE Hamiltonian for testing.
    
    This version runs without sharding for verification on small systems.
    
    Uses FFT convolution following load_wfns.py pattern with norm='forward':
    - Convolution gives automatic 1/Nk factor
    - V term uses explicit 1/Nk
    
    Args:
        X: (nb_trial, nc, nv, nk) trial vectors
        psi_c: (nk, nc, nspinor, n_rmu) conduction wfns
        psi_v: (nk, nv, nspinor, n_rmu) valence wfns
        eps_c: (nk, nc) conduction energies
        eps_v: (nk, nv) valence energies
        W_q: (n_rmu, n_rmu, nkx, nky, nkz) screened exchange in q-space
        V_q0: (n_rmu, n_rmu) bare Coulomb at q=0
        nkx, nky, nkz: k-grid dimensions
        
    Returns:
        HX: (nb_trial, nc, nv, nk)
    """
    nk = nkx * nky * nkz
    nb_trial = X.shape[0]
    
    # ===== D term: local =====
    # Note: eps are (nk, nb), need to broadcast correctly
    # D(c,v,k) = ε_c(k,c) - ε_v(k,v)
    # X is (b, c, v, k) so we need (1, nc, 1, nk) - (1, 1, nv, nk)
    delta_E = energy_diff_cv_k(eps_c, eps_v)[None, :, :, :]
    D_term = delta_E * X
    
    # ===== Pair amplitude: M(k, c, v, μ) = Σ_s ψ*_{c,s}(μ,k) ψ_{v,s}(μ,k) =====
    M = compute_pair_amplitude(psi_c, psi_v)  # (nk, nc, nv, n_rmu)
    
    # ===== V term: q=0 only =====
    # S(b, ν, k) = Σ_{c'v'} M(k, c', v', ν) X(b, c', v', k)
    S_V = jnp.einsum('kcvN,bcvk->bNk', M, X)

    # Split the overall 1/Nk BZ prefactor symmetrically as 1/sqrt(Nk) on encode and decode.
    sqrt_nk = jnp.sqrt(jnp.asarray(nk, dtype=jnp.float64))
    S_V = S_V / sqrt_nk
    U_V = jnp.einsum('MN,bNk->bMk', V_q0, S_V)
    V_term = jnp.einsum('kcvM,bMk->bcvk', jnp.conj(M), U_V) / sqrt_nk
    
    # ===== W term: FFT convolution (Henneke eq (4-6), spin-matrix form) =====
    # Build T(b, μ, ν, t, s, k) = Σ_{c',v'} ψ_c(k,c',t,μ) * ψ*_v(k,v',s,ν) * X(b,c',v',k)
    n_rmu = psi_c.shape[-1]
    nspinor = psi_c.shape[2]

    # R(b, c, k, s, ν) = Σ_v conj(ψ_v(k,v,s,ν)) * X(b,c,v,k)
    R = jnp.einsum('kvsN,bcvk->bcksN', jnp.conj(psi_v), X)
    # T(b, μ, ν, t, s, k) = Σ_c ψ_c(k,c,t,μ) * R(b,c,k,s,ν)
    T = jnp.einsum('kctM,bcksN->bMNtsk', psi_c, R)

    # Convolution in k for each (μ,ν,t,s) using unitary FFTs
    T_k = T.reshape(nb_trial, n_rmu, n_rmu, nspinor, nspinor, nkx, nky, nkz)
    T_R = jnp.fft.ifftn(T_k, axes=(5, 6, 7), norm='ortho')
    W_R = jnp.fft.ifftn(W_q, axes=(2, 3, 4), norm='ortho')  # (μ,ν,nkx,nky,nkz)
    U_R = W_R[None, :, :, None, None, :, :, :] * T_R
    U_q = jnp.fft.fftn(U_R, axes=(5, 6, 7), norm='ortho')
    U = U_q.reshape(nb_trial, n_rmu, n_rmu, nspinor, nspinor, nk)

    # Decode: WX(b,c,v,k) = Σ_{μ,ν,t,s} ψ*_c(k,c,t,μ) * U(b,μ,ν,t,s,k) * ψ_v(k,v,s,ν)
    A = jnp.einsum('kctM,bMNtsk->bcNsk', jnp.conj(psi_c), U)
    W_term = jnp.einsum('kvsN,bcNsk->bcvk', psi_v, A) / sqrt_nk
    
    # For spinors: H = D + V - W (no factor of 2 on V)
    # V and W couple to charge density at each vertex, which is spin-traced: ρ = Σ_σ ψ*_σ ψ_σ
    return D_term + V_term - W_term


@jax.jit
def apply_bse_hamiltonian_single_device_jit(
    X: jax.Array,
    psi_c: jax.Array,
    psi_v: jax.Array,
    eps_c: jax.Array,
    eps_v: jax.Array,
    W_q: jax.Array,
    V_q0: jax.Array,
    nkx: int,
    nky: int,
    nkz: int,
) -> jax.Array:
    """JIT-compiled version for single device."""
    return apply_bse_hamiltonian_single_device(
        X, psi_c, psi_v, eps_c, eps_v, W_q, V_q0, nkx, nky, nkz
    )


# ============== Lanczos eigensolver ==============

def block_lanczos_eig(
    matvec: Callable[[jax.Array], jax.Array],
    shape: Tuple[int, ...],
    n_eig: int = 20,
    block_size: int = 4,
    max_iter: int = 50,
    tol: float = 1e-8,
    seed: int = 42,
) -> Tuple[jax.Array, jax.Array]:
    """Block Lanczos algorithm for lowest eigenvalues of BSE Hamiltonian.
    
    Uses a block of trial vectors for faster convergence and better
    parallelism over multiple eigenvalues.
    
    Args:
        matvec: Function X -> HX where X has shape (block_size, *shape)
        shape: Shape of a single trial vector (nc, nv, nk)
        n_eig: Number of lowest eigenvalues to compute
        block_size: Number of vectors processed together
        max_iter: Maximum Lanczos iterations
        tol: Convergence tolerance for eigenvalue change
        seed: Random seed for initial vectors
        
    Returns:
        eigenvalues: (n_eig,) lowest eigenvalues
        eigenvectors: (n_eig, *shape) corresponding eigenvectors
    """
    n_flat = np.prod(shape)
    key = jax.random.PRNGKey(seed)
    
    # Initialize random block of starting vectors
    k1, k2 = jax.random.split(key)
    Q0 = jax.random.normal(k1, (block_size, *shape), dtype=jnp.float64)
    Q0 = Q0 + 1j * jax.random.normal(k2, (block_size, *shape), dtype=jnp.float64)
    
    # Flatten for orthogonalization
    Q0_flat = Q0.reshape(block_size, n_flat)
    Q0_flat, _ = jnp.linalg.qr(Q0_flat.T)  # (n_flat, block_size)
    Q0_flat = Q0_flat.T  # (block_size, n_flat)
    
    # Storage for Lanczos vectors and tridiagonal matrix elements
    # Using Python list for dynamic accumulation (converted to array for eigh)
    Q_blocks = [Q0_flat]
    alpha_blocks = []  # Diagonal blocks
    beta_blocks = []   # Off-diagonal blocks
    
    Q_current = Q0_flat.reshape(block_size, *shape)
    
    for j in range(max_iter):
        # Apply Hamiltonian
        Z = matvec(Q_current)  # (block_size, *shape)
        Z_flat = Z.reshape(block_size, n_flat)
        Q_current_flat = Q_current.reshape(block_size, n_flat)
        
        # Compute alpha_j = Q_j^H @ Z
        alpha_j = Q_current_flat.conj() @ Z_flat.T  # (block_size, block_size)
        alpha_blocks.append(alpha_j)
        
        # Orthogonalize against previous block
        Z_flat = Z_flat - alpha_j.T @ Q_current_flat
        
        if j > 0:
            Q_prev_flat = Q_blocks[-2]
            Z_flat = Z_flat - beta_blocks[-1].T @ Q_prev_flat
        
        # Full reorthogonalization against all previous vectors
        for Q_old in Q_blocks:
            proj = Z_flat @ Q_old.conj().T  # (block_size, block_size)
            Z_flat = Z_flat - proj @ Q_old
        
        # QR factorization for next block
        Z_flat_T, R = jnp.linalg.qr(Z_flat.T)  # Z_flat_T: (n_flat, block_size)
        beta_j = R.T  # (block_size, block_size)
        beta_blocks.append(beta_j)
        
        # Check for convergence (small beta)
        beta_norm = jnp.linalg.norm(beta_j)
        if beta_norm < tol * block_size:
            print(f"Block Lanczos converged at iteration {j+1}")
            break
        
        Q_next_flat = Z_flat_T.T  # (block_size, n_flat)
        Q_blocks.append(Q_next_flat)
        Q_current = Q_next_flat.reshape(block_size, *shape)
    
    # Build block tridiagonal matrix T
    n_blocks = len(alpha_blocks)
    T_size = n_blocks * block_size
    T = jnp.zeros((T_size, T_size), dtype=jnp.complex128)
    
    for i, alpha in enumerate(alpha_blocks):
        start = i * block_size
        end = (i + 1) * block_size
        T = T.at[start:end, start:end].set(alpha)
        
        if i < len(beta_blocks) - 1:
            beta = beta_blocks[i]
            T = T.at[end:end+block_size, start:end].set(beta)
            T = T.at[start:end, end:end+block_size].set(beta.conj().T)
    
    # Ensure Hermitian
    T = (T + T.conj().T) / 2
    
    # Diagonalize T
    evals_T, vecs_T = jnp.linalg.eigh(T)
    
    # Select lowest eigenvalues
    idx = jnp.argsort(evals_T.real)[:n_eig]
    eigenvalues = evals_T[idx].real
    
    # Reconstruct eigenvectors in original space
    Q_all = jnp.concatenate(Q_blocks[:n_blocks], axis=0)  # (n_blocks*block_size, n_flat)
    eigenvectors_flat = vecs_T[:, idx].T @ Q_all  # (n_eig, n_flat)
    eigenvectors = eigenvectors_flat.reshape(n_eig, *shape)
    
    # Normalize
    norms = jnp.linalg.norm(eigenvectors.reshape(n_eig, -1), axis=1, keepdims=True)
    eigenvectors = eigenvectors / norms.reshape(n_eig, *([1] * len(shape)))
    
    return eigenvalues, eigenvectors


def simple_lanczos_eig(
    matvec: Callable[[jax.Array], jax.Array],
    n: int,
    n_eig: int = 20,
    max_iter: int = 100,
    seed: int = 42,
) -> Tuple[jax.Array, jax.Array]:
    """Simple Lanczos algorithm for lowest eigenvalues (Python loop version).
    
    Single-vector version for comparison/debugging.
    Slower than lanczos_eig_jit but easier to debug.
    
    Args:
        matvec: Function v -> Hv for flattened vectors
        n: Dimension of the problem
        n_eig: Number of lowest eigenvalues to compute
        max_iter: Maximum iterations
        seed: Random seed
        
    Returns:
        eigenvalues: (n_eig,) lowest eigenvalues  
        eigenvectors: (n_eig, n) corresponding eigenvectors
    """
    key = jax.random.PRNGKey(seed)
    k1, k2 = jax.random.split(key)
    
    q = jax.random.normal(k1, (n,), dtype=jnp.float64)
    q = q + 1j * jax.random.normal(k2, (n,), dtype=jnp.float64)
    q = q / jnp.linalg.norm(q)
    
    Q = jnp.zeros((n, max_iter + 1), dtype=jnp.complex128)
    Q = Q.at[:, 0].set(q)
    alpha = jnp.zeros((max_iter,), dtype=jnp.float64)
    beta = jnp.zeros((max_iter,), dtype=jnp.float64)
    
    for j in range(max_iter):
        z = matvec(q)
        alpha = alpha.at[j].set(jnp.vdot(q, z).real)
        
        if j > 0:
            z = z - beta[j-1] * Q[:, j-1]
        z = z - alpha[j] * q
        
        # Full reorthogonalization
        for i in range(j + 1):
            proj = jnp.vdot(Q[:, i], z)
            z = z - proj * Q[:, i]
        
        beta = beta.at[j].set(jnp.linalg.norm(z))
        
        if beta[j] < 1e-12:
            max_iter = j + 1
            break
            
        q = z / beta[j]
        Q = Q.at[:, j + 1].set(q)
    
    # Build tridiagonal matrix
    T = jnp.diag(alpha[:max_iter])
    if max_iter > 1:
        off = beta[:max_iter-1]
        T = T + jnp.diag(off, 1) + jnp.diag(off, -1)
    
    evals_T, vecs_T = jnp.linalg.eigh(T)
    idx = jnp.argsort(evals_T)[:n_eig]
    
    eigenvalues = evals_T[idx]
    eigenvectors = (Q[:, :max_iter] @ vecs_T[:, idx]).T
    
    # Normalize
    norms = jnp.linalg.norm(eigenvectors, axis=1, keepdims=True)
    eigenvectors = eigenvectors / norms
    
    return eigenvalues, eigenvectors


def lanczos_eig_jit(
    matvec: Callable[[jax.Array], jax.Array],
    n: int,
    n_eig: int = 20,
    max_iter: int = 100,
    seed: int = 42,
    n_reorth: int = 2,
) -> Tuple[jax.Array, jax.Array]:
    """JIT-able Lanczos algorithm using lax.fori_loop.
    
    This version is fully JIT-compatible and avoids Python control flow
    in the inner loop. Uses fixed-size pre-allocated arrays.
    
    The algorithm:
    1. Pre-allocate Q (n, max_iter), alpha (max_iter), beta (max_iter)
    2. Use lax.fori_loop for the main iteration
    3. Use selective reorthogonalization (cheaper than full)
    4. Build tridiagonal T and solve with jnp.linalg.eigh
    
    Memory: O(n × max_iter) for Q matrix
    - For n = 540000 (50×50×216), max_iter = 100: ~850 MB
    - This fits comfortably on a single GPU
    
    Args:
        matvec: JIT-compiled function v -> Hv for flattened vectors
        n: Dimension of the problem
        n_eig: Number of lowest eigenvalues to compute
        max_iter: Maximum iterations (pre-allocated)
        seed: Random seed
        n_reorth: Reorthogonalize against this many recent vectors
                  (set to max_iter for full, 2 for classic 3-term)
        
    Returns:
        eigenvalues: (n_eig,) lowest eigenvalues  
        eigenvectors: (n_eig, n) corresponding eigenvectors
    """
    key = jax.random.PRNGKey(seed)
    k1, k2 = jax.random.split(key)
    
    # Initialize random starting vector
    q0 = jax.random.normal(k1, (n,), dtype=jnp.float64)
    q0 = q0 + 1j * jax.random.normal(k2, (n,), dtype=jnp.float64)
    q0 = q0 / jnp.linalg.norm(q0)
    
    # Pre-allocate all arrays for JIT compatibility
    Q = jnp.zeros((n, max_iter), dtype=jnp.complex128)
    Q = Q.at[:, 0].set(q0)
    alpha = jnp.zeros((max_iter,), dtype=jnp.float64)
    beta = jnp.zeros((max_iter,), dtype=jnp.float64)
    
    # Lanczos iteration body - must be pure for lax.fori_loop
    def lanczos_step(j, carry):
        Q, alpha, beta, q_prev = carry
        
        # Apply Hamiltonian
        z = matvec(q_prev)
        
        # Compute alpha[j] = <q|H|q>
        alpha_j = jnp.vdot(q_prev, z).real
        alpha = alpha.at[j].set(alpha_j)
        
        # Orthogonalize: z = z - alpha[j]*q - beta[j-1]*q_{j-1}
        z = z - alpha_j * q_prev
        
        # Subtract previous vector contribution (when j > 0)
        q_prev_prev = Q[:, jnp.maximum(j - 1, 0)]
        beta_prev = jnp.where(j > 0, beta[j - 1], 0.0)
        z = z - beta_prev * q_prev_prev
        
        # Selective reorthogonalization against recent vectors
        # This is a compromise between full reorth (expensive) and none (unstable)
        def reorth_body(i, z_acc):
            # Only reorthogonalize against valid vectors (i < j)
            valid = i < j
            q_i = Q[:, i]
            proj = jnp.where(valid, jnp.vdot(q_i, z_acc), 0.0+0j)
            return z_acc - proj * q_i
        
        # Reorthogonalize against last n_reorth vectors
        start_idx = jnp.maximum(0, j - n_reorth)
        z = lax.fori_loop(start_idx, j + 1, reorth_body, z)
        
        # Compute beta[j] = ||z||
        beta_j = jnp.linalg.norm(z)
        beta = beta.at[j].set(beta_j)
        
        # Normalize to get next q (with safeguard for breakdown)
        q_next = z / jnp.maximum(beta_j, 1e-15)
        
        # Store in Q matrix for next iteration
        Q = Q.at[:, jnp.minimum(j + 1, max_iter - 1)].set(q_next)
        
        return (Q, alpha, beta, q_next)
    
    # Run Lanczos iterations
    init_carry = (Q, alpha, beta, q0)
    Q, alpha, beta, _ = lax.fori_loop(0, max_iter, lanczos_step, init_carry)
    
    # Build symmetric tridiagonal matrix T
    # T[i,i] = alpha[i], T[i,i+1] = T[i+1,i] = beta[i]
    T = jnp.diag(alpha)
    off_diag = beta[:-1]
    T = T + jnp.diag(off_diag, 1) + jnp.diag(off_diag, -1)
    
    # Solve tridiagonal eigenproblem (very fast for size max_iter)
    evals_T, vecs_T = jnp.linalg.eigh(T)
    
    # Select lowest n_eig eigenvalues
    idx = jnp.argsort(evals_T)[:n_eig]
    eigenvalues = evals_T[idx]
    
    # Reconstruct eigenvectors: psi = Q @ y where y are Ritz vectors
    eigenvectors = (Q @ vecs_T[:, idx]).T  # (n_eig, n)
    
    # Normalize eigenvectors
    norms = jnp.linalg.norm(eigenvectors, axis=1, keepdims=True)
    eigenvectors = eigenvectors / jnp.maximum(norms, 1e-15)
    
    return eigenvalues, eigenvectors


# ============== Convenience wrapper ==============

def solve_bse(
    psi_c: jax.Array,
    psi_v: jax.Array,
    eps_c: jax.Array,
    eps_v: jax.Array,
    W_q: jax.Array,
    V_q0: jax.Array,
    nkx: int,
    nky: int,
    nkz: int,
    n_eig: int = 20,
    max_iter: int = 100,
    use_block: bool = False,
    block_size: int = 4,
    use_jit_lanczos: bool = True,
    n_reorth: int = 10,
) -> Tuple[jax.Array, jax.Array]:
    """Solve BSE for lowest exciton eigenvalues.
    
    High-level interface that sets up the matvec and runs Lanczos.
    
    Args:
        psi_c: (nk, nc, nspinor, n_rmu) conduction wavefunctions
        psi_v: (nk, nv, nspinor, n_rmu) valence wavefunctions
        eps_c: (nk, nc) conduction band energies
        eps_v: (nk, nv) valence band energies
        W_q: (n_rmu, n_rmu, nkx, nky, nkz) screened exchange in q-space
        V_q0: (n_rmu, n_rmu) bare Coulomb at q=0
        nkx, nky, nkz: k-grid dimensions
        n_eig: Number of lowest exciton states
        max_iter: Maximum Lanczos iterations
        use_block: Use block Lanczos (faster for many eigenvalues)
        block_size: Block size for block Lanczos
        use_jit_lanczos: Use JIT-compiled Lanczos (faster, default True)
        n_reorth: Number of vectors to reorthogonalize against (for JIT version)
        
    Returns:
        eigenvalues: (n_eig,) exciton energies
        eigenvectors: Exciton wavefunctions A_cvk
    """
    nk, nc, nspinor, n_rmu = psi_c.shape
    nv = psi_v.shape[1]
    shape = (nc, nv, nk)
    n_flat = nc * nv * nk
    
    # JIT-compile the single-device matvec with captured arrays
    # Using partial to create a closure that JIT can trace
    @partial(jax.jit, static_argnames=('nkx', 'nky', 'nkz'))
    def _matvec_impl(v, psi_c, psi_v, eps_c, eps_v, W_q, V_q0, nkx, nky, nkz):
        X = v.reshape(1, nc, nv, nk)
        HX = apply_bse_hamiltonian_single_device(
            X, psi_c, psi_v, eps_c, eps_v, W_q, V_q0, nkx, nky, nkz
        )
        return HX.reshape(-1)
    
    # Create matvec with captured data arrays
    def matvec_flat(v):
        return _matvec_impl(v, psi_c, psi_v, eps_c, eps_v, W_q, V_q0, nkx, nky, nkz)
    
    if use_block:
        @jax.jit
        def matvec_block(X):
            return apply_bse_hamiltonian_single_device(
                X, psi_c, psi_v, eps_c, eps_v, W_q, V_q0, nkx, nky, nkz
            )
        
        eigenvalues, eigenvectors = block_lanczos_eig(
            matvec_block, shape, n_eig=n_eig, block_size=block_size, max_iter=max_iter
        )
    elif use_jit_lanczos:
        # Use JIT-compiled Lanczos with lax.fori_loop
        eigenvalues, eigenvectors = lanczos_eig_jit(
            matvec_flat, n_flat, n_eig=n_eig, max_iter=max_iter, n_reorth=n_reorth
        )
        eigenvectors = eigenvectors.reshape(n_eig, *shape)
    else:
        # Use Python-loop Lanczos (easier to debug)
        eigenvalues, eigenvectors = simple_lanczos_eig(
            matvec_flat, n_flat, n_eig=n_eig, max_iter=max_iter
        )
        eigenvectors = eigenvectors.reshape(n_eig, *shape)
    
    return eigenvalues, eigenvectors


def ring_matvec_smoke_test(px: int = 2, py: int = 2) -> None:
    """Small CPU-mesh smoke test for ring-based matvec and shardings."""
    devices = jax.devices()
    if len(devices) < px * py:
        raise RuntimeError(
            f"Need {px*py} devices, found {len(devices)}. "
            "Set XLA_FLAGS=--xla_force_host_platform_device_count=... before running."
        )
    mesh = Mesh(np.array(devices[:px * py]).reshape(px, py), axis_names=("x", "y"))
    sh = make_bse_shardings(mesh)

    nkx, nky, nkz = 2, 2, 1
    nk = nkx * nky * nkz
    nc, nv, nspinor, n_rmu = 4 * px, 4 * py, 2, 8 * px * py

    key = jax.random.PRNGKey(0)
    psi_c = jax.random.normal(key, (nk, nc, nspinor, n_rmu)) + 1j * jax.random.normal(key, (nk, nc, nspinor, n_rmu))
    psi_v = jax.random.normal(key, (nk, nv, nspinor, n_rmu)) + 1j * jax.random.normal(key, (nk, nv, nspinor, n_rmu))
    eps_c = jax.random.uniform(key, (nk, nc), minval=0.1, maxval=0.5)
    eps_v = jax.random.uniform(key, (nk, nv), minval=-0.5, maxval=-0.1)
    W_q = jax.random.normal(key, (n_rmu, n_rmu, nkx, nky, nkz)) * 0.01
    V_q0 = jnp.eye(n_rmu) * 0.05
    X = jax.random.normal(key, (1, nc, nv, nk)) + 1j * jax.random.normal(key, (1, nc, nv, nk))

    with mesh:
        psi_c_X = jax.lax.with_sharding_constraint(psi_c, sh.psi_x)
        psi_c_Y = jax.lax.with_sharding_constraint(psi_c, sh.psi_y)
        psi_v_X = jax.lax.with_sharding_constraint(psi_v, sh.psi_x)
        psi_v_Y = jax.lax.with_sharding_constraint(psi_v, sh.psi_y)
        W_q = jax.lax.with_sharding_constraint(W_q, sh.W)
        V_q0 = jax.lax.with_sharding_constraint(V_q0, sh.V)
        X = jax.lax.with_sharding_constraint(X, sh.X)

        matvec = build_bse_ring_matvec(mesh, nkx, nky, nkz)
        HX = matvec(X, psi_c_X, psi_c_Y, psi_v_X, psi_v_Y, eps_c, eps_v, W_q, V_q0)
        HX.block_until_ready()

    print(f"HX sharding: {HX.sharding}")
    if hasattr(jax.debug, "inspect_array_sharding"):
        try:
            jax.debug.inspect_array_sharding(HX, name="HX")
        except TypeError:
            try:
                jax.debug.inspect_array_sharding(HX, callback=lambda *_: None)
            except TypeError:
                pass


def _find_restart_file(input_file: str) -> str:
    input_dir = os.path.dirname(os.path.abspath(input_file))
    candidates = []
    candidates.extend(sorted(glob.glob(os.path.join(input_dir, "tmp", "isdf_tensors_*.h5"))))
    candidates.extend(sorted(glob.glob(os.path.join(input_dir, "isdf_tensors_*.h5"))))
    candidates.extend([
        os.path.join(input_dir, "tmp", "taggedarrays600.h5"),
        os.path.join(input_dir, "tmp", "taggedarrays.h5"),
        os.path.join(input_dir, "taggedarrays.h5"),
    ])
    for path in candidates:
        if os.path.exists(path):
            return path
    raise FileNotFoundError(f"Could not find restart file in {input_dir}")


def ring_matvec_correctness_check(
    input_file: str,
    n_val: int = 4,
    n_cond: int = 4,
    px: int = 2,
    py: int = 2,
    component_check: bool = False,
) -> None:
    restart_file = _find_restart_file(input_file)
    devices = jax.devices()
    if len(devices) < px * py:
        raise RuntimeError(
            f"Need {px*py} devices, found {len(devices)}. "
            "Set XLA_FLAGS=--xla_force_host_platform_device_count=... before running."
        )
    mesh = Mesh(np.array(devices[:px * py]).reshape(px, py), axis_names=("x", "y"))
    sh = make_bse_shardings(mesh)

    with h5py.File(restart_file, "r") as f:
        V_qmunu = jnp.asarray(f["V_qmunu"][:])
        if "W0_qmunu" in f and bool(f["W0_qmunu"].attrs.get("W0_ready", False)):
            W0_qmunu = jnp.asarray(f["W0_qmunu"][:])
        else:
            W0_qmunu = None
        psi_l = jnp.asarray(f["psi_l"][:])
        psi_r = jnp.asarray(f["psi_r"][:])
        enk_l = jnp.asarray(f["enk_l"][:])
        enk_r = jnp.asarray(f["enk_r"][:])

    nkx, nky, nkz = V_qmunu.shape[3:6]
    nk = nkx * nky * nkz
    n_rmu = int(V_qmunu.shape[-1])
    lcm_xy = math.lcm(px, py)
    n_rmu_pad = ((n_rmu + lcm_xy - 1) // lcm_xy) * lcm_xy

    mean_enk_l = jnp.mean(enk_l, axis=0)
    mean_enk_r = jnp.mean(enk_r, axis=0)
    val_mask_l = mean_enk_l < 0.0
    cond_mask_r = mean_enk_r > 0.0
    n_val = min(n_val, int(jnp.sum(val_mask_l)))
    n_cond = min(n_cond, int(jnp.sum(cond_mask_r)))
    val_indices = jnp.argsort(jnp.where(val_mask_l, mean_enk_l, -jnp.inf))[-n_val:]
    cond_indices = jnp.argsort(jnp.where(cond_mask_r, mean_enk_r, jnp.inf))[:n_cond]

    psi_v = psi_l[:, val_indices, :, :]
    psi_c = psi_r[:, cond_indices, :, :]
    eps_v = enk_l[:, val_indices]
    eps_c = enk_r[:, cond_indices]

    psi_v = _pad_last_axis(psi_v, n_rmu_pad)
    psi_c = _pad_last_axis(psi_c, n_rmu_pad)
    psi_v, n_val_pad = _pad_axis_to_multiple(psi_v, axis=1, multiple=py)
    psi_c, n_cond_pad = _pad_axis_to_multiple(psi_c, axis=1, multiple=px)
    eps_v, _ = _pad_axis_to_multiple(eps_v, axis=1, multiple=py)
    eps_c, _ = _pad_axis_to_multiple(eps_c, axis=1, multiple=px)
    V_q0 = V_qmunu[0, 0, 0, 0, 0, 0, :, :]
    V_q0 = _pad_last_two_axes(V_q0, n_rmu_pad)
    W_src = W0_qmunu if W0_qmunu is not None else V_qmunu
    W_q = W_src[0, 0, 0, :, :, :, :, :].transpose(3, 4, 0, 1, 2)
    W_q = _pad_first_two_axes(W_q, n_rmu_pad)

    key = jax.random.PRNGKey(0)
    X = jax.random.normal(key, (1, n_cond_pad, n_val_pad, nk)) + 1j * jax.random.normal(key, (1, n_cond_pad, n_val_pad, nk))

    HX_ref = apply_bse_hamiltonian_single_device(
        X, psi_c, psi_v, eps_c, eps_v, W_q, V_q0, nkx, nky, nkz
    )

    with mesh:
        psi_c_X = jax.lax.with_sharding_constraint(psi_c, sh.psi_x)
        psi_c_Y = jax.lax.with_sharding_constraint(psi_c, sh.psi_y)
        psi_v_X = jax.lax.with_sharding_constraint(psi_v, sh.psi_x)
        psi_v_Y = jax.lax.with_sharding_constraint(psi_v, sh.psi_y)
        W_q = jax.lax.with_sharding_constraint(W_q, sh.W)
        V_q0 = jax.lax.with_sharding_constraint(V_q0, sh.V)
        X = jax.lax.with_sharding_constraint(X, sh.X)

        matvec = build_bse_ring_matvec(mesh, nkx, nky, nkz)
        HX_ring = matvec(X, psi_c_X, psi_c_Y, psi_v_X, psi_v_Y, eps_c, eps_v, W_q, V_q0)
        HX_ring.block_until_ready()

    HX_ring_host = jax.device_get(HX_ring)
    diff = jnp.linalg.norm(HX_ring_host - HX_ref) / jnp.maximum(jnp.linalg.norm(HX_ref), 1e-12)
    print(f"Relative error ||HX_ring - HX_ref||/||HX_ref||: {float(diff):.3e}")

    if component_check:
        sqrt_nk = jnp.sqrt(jnp.asarray(nk, dtype=jnp.float64))
        M = compute_pair_amplitude(psi_c, psi_v)
        D_ref = apply_D(X, eps_c, eps_v)
        S_V = jnp.einsum('kcvN,bcvk->bNk', M, X) / sqrt_nk
        U_V = jnp.einsum('MN,bNk->bMk', V_q0, S_V)
        V_ref = jnp.einsum('kcvM,bMk->bcvk', jnp.conj(M), U_V) / sqrt_nk

        R = jnp.einsum('kvsN,bcvk->bcksN', jnp.conj(psi_v), X)
        T = jnp.einsum('kctM,bcksN->bMNtsk', psi_c, R)
        T_k = T.reshape(X.shape[0], n_rmu_pad, n_rmu_pad, psi_c.shape[2], psi_c.shape[2], nkx, nky, nkz)
        T_R = jnp.fft.ifftn(T_k, axes=(5, 6, 7), norm='ortho')
        W_R = jnp.fft.ifftn(W_q, axes=(2, 3, 4), norm='ortho')
        U_R = W_R[None, :, :, None, None, :, :, :] * T_R
        U_q = jnp.fft.fftn(U_R, axes=(5, 6, 7), norm='ortho')
        U = U_q.reshape(X.shape[0], n_rmu_pad, n_rmu_pad, psi_c.shape[2], psi_c.shape[2], nk)
        A = jnp.einsum('kctM,bMNtsk->bcNsk', jnp.conj(psi_c), U)
        W_ref = jnp.einsum('kvsN,bcNsk->bcvk', psi_v, A) / sqrt_nk

        comp_matvec = build_bse_ring_matvec(mesh, nkx, nky, nkz)
        with mesh:
            D_ring = apply_D(X, eps_c, eps_v)
            V_ring = comp_matvec(
                X, psi_c_X, psi_c_Y, psi_v_X, psi_v_Y, eps_c * 0.0, eps_v * 0.0, W_q * 0.0, V_q0
            )
            W_ring = comp_matvec(
                X, psi_c_X, psi_c_Y, psi_v_X, psi_v_Y, eps_c * 0.0, eps_v * 0.0, W_q, V_q0 * 0.0
            )
            D_ring.block_until_ready()
            V_ring.block_until_ready()
            W_ring.block_until_ready()

        def _rel_err(a, b):
            return float(jnp.linalg.norm(a - b) / jnp.maximum(jnp.linalg.norm(b), 1e-12))

        print(f"Component error D: { _rel_err(D_ring, D_ref):.3e}")
        print(f"Component error V: { _rel_err(V_ring, V_ref):.3e}")
        print(f"Component error W: { _rel_err(-W_ring, W_ref):.3e}")


if __name__ == "__main__":
    if "--ring-check" in sys.argv:
        import argparse

        parser = argparse.ArgumentParser(description="Ring matvec correctness check")
        parser.add_argument("-i", "--input", required=True, help="COHSEX input file (for restart lookup)")
        parser.add_argument("--n-val", type=int, default=4)
        parser.add_argument("--n-cond", type=int, default=4)
        parser.add_argument("--px", type=int, default=2)
        parser.add_argument("--py", type=int, default=2)
        args, _ = parser.parse_known_args()
        ring_matvec_correctness_check(args.input, args.n_val, args.n_cond, args.px, args.py, "--components" in sys.argv)
        raise SystemExit(0)

    if "--ring-test" in sys.argv:
        ring_matvec_smoke_test()
        raise SystemExit(0)

    # Quick sanity check with random data
    print("Testing BSE matvec with random data...")
    
    nk, nc, nv, nspinor, n_rmu = 8, 4, 4, 2, 32
    nkx, nky, nkz = 2, 2, 2
    
    key = jax.random.PRNGKey(0)
    keys = jax.random.split(key, 7)
    
    psi_c = jax.random.normal(keys[0], (nk, nc, nspinor, n_rmu)) + \
            1j * jax.random.normal(keys[1], (nk, nc, nspinor, n_rmu))
    psi_v = jax.random.normal(keys[2], (nk, nv, nspinor, n_rmu)) + \
            1j * jax.random.normal(keys[3], (nk, nv, nspinor, n_rmu))
    
    # Physical energies: valence < 0 < conduction (gap ~ 1 eV)
    eps_v = jax.random.uniform(keys[4], (nk, nv), minval=-0.5, maxval=-0.1)
    eps_c = jax.random.uniform(keys[5], (nk, nc), minval=0.1, maxval=0.5)
    
    # Random W_q and V_q0 for testing (small values for physical eigenvalues)
    W_q = jax.random.normal(keys[6], (n_rmu, n_rmu, nkx, nky, nkz)) * 0.01
    V_q0 = jnp.eye(n_rmu) * 0.05
    
    # Test single trial vector
    X = jnp.ones((1, nc, nv, nk), dtype=jnp.complex128)
    X = X / jnp.linalg.norm(X)
    
    HX = apply_bse_hamiltonian_single_device(
        X, psi_c, psi_v, eps_c, eps_v, W_q, V_q0, nkx, nky, nkz
    )
    print(f"Input shape: {X.shape}, Output shape: {HX.shape}")
    E_expect = jnp.vdot(X.flatten(), HX.flatten()).real
    ryd2ev = 13.6056980659
    print(f"Expectation value: {E_expect:.6f} Ry = {E_expect * ryd2ev:.4f} eV")
    
    # Test Lanczos solver
    print("\nRunning Lanczos solver...")
    eigenvalues, eigenvectors = solve_bse(
        psi_c, psi_v, eps_c, eps_v, W_q, V_q0, nkx, nky, nkz,
        n_eig=5, max_iter=30
    )
    ryd2ev = 13.6056980659
    print(f"Lowest 5 eigenvalues (Ry): {eigenvalues}")
    print(f"Lowest 5 eigenvalues (eV): {eigenvalues * ryd2ev}")
