"""Distributed BSE kernels (ring collectives and sharded matvecs)."""
from __future__ import annotations

from types import SimpleNamespace
from typing import Optional

import jax
import jax.numpy as jnp
import numpy as np
from jax import lax
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

try:
    from jax import shard_map as _shard_map_fn
except ImportError:  # pragma: no cover - older JAX
    from jax.experimental import shard_map as _shard_map_mod
    _shard_map_fn = _shard_map_mod.shard_map

import common.timing as timing
from common.fft_helpers import (
    make_sharded_fftn_3d,
    make_sharded_ifftn_3d,
)
from .bse_io import _find_restart_file, _load_ring_subset, load_bse_data_from_restart_sharded
from .bse_serial import apply_D, apply_bse_hamiltonian_single_device, compute_pair_amplitude


jax.config.update("jax_enable_x64", True)


def create_mesh_2d(devices: Optional[list] = None) -> Mesh:
    """Create a 2D device mesh for BSE sharding."""
    if devices is None:
        devices = jax.devices()

    n_devices = len(devices)
    px = int(np.sqrt(n_devices))
    while n_devices % px != 0:
        px -= 1
    py = n_devices // px

    device_array = np.array(devices).reshape(px, py)
    return Mesh(device_array, axis_names=("x", "y"))


def make_bse_shardings(mesh_xy: Mesh) -> SimpleNamespace:
    return SimpleNamespace(
        X=NamedSharding(mesh_xy, P(None, "x", "y", None)),
        X_full=NamedSharding(mesh_xy, P(None, None, "x", "y", None)),
        psi_x=NamedSharding(mesh_xy, P(None, None, None, "x")),
        psi_y=NamedSharding(mesh_xy, P(None, None, None, "y")),
        V=NamedSharding(mesh_xy, P("x", "y")),
        W=NamedSharding(mesh_xy, P("x", "y", None, None, None)),
        eps=NamedSharding(mesh_xy, P(None, None)),
        R=NamedSharding(mesh_xy, P(None, "x", None, None, "y")),
        T=NamedSharding(mesh_xy, P(None, "x", "y", None, None, None)),
        U=NamedSharding(mesh_xy, P(None, "x", "y", None, None, None)),
        A=NamedSharding(mesh_xy, P(None, "x", "y", None, None)),
        D=NamedSharding(mesh_xy, P(None, "x", "y", None, None)),
        S=NamedSharding(mesh_xy, P(None, "y", None)),
        U_mu=NamedSharding(mesh_xy, P(None, "x", None)),
        d_mu=NamedSharding(mesh_xy, P(None, "x")),
    )


def _ring_perm(axis_size: int) -> tuple[tuple[int, int], ...]:
    return tuple((i, (i + 1) % axis_size) for i in range(axis_size))


def _ring_sum_valence(
    X: jax.Array,
    psi_v_Y: jax.Array,
    v_chunk: int,
    py: int,
    nu_local: int,
) -> jax.Array:
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

    _, R_total = lax.fori_loop(0, py, step, (X, R0))
    return R_total


def _ring_sum_conduction(
    R: jax.Array,
    psi_c_X: jax.Array,
    c_chunk: int,
    px: int,
    mu_local: int,
) -> jax.Array:
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

    _, T_total = lax.fori_loop(0, px, step, (R, T0))
    return T_total


def _ring_sum_conduction_first(
    X: jax.Array,
    psi_c_Y: jax.Array,
    c_chunk: int,
    px: int,
    nu_local: int,
) -> jax.Array:
    """Sum over conduction first: R(v,k,s,nu) = sum_c psi_c^*(nu) X(c,v,k)."""
    axis_index_x = jnp.asarray(lax.axis_index("x"), dtype=jnp.int32)
    nk, _, nspinor, _ = psi_c_Y.shape
    v_chunk = X.shape[2]

    R0 = jnp.zeros((X.shape[0], v_chunk, nk, nspinor, nu_local), dtype=X.dtype)
    perm = _ring_perm(px)

    def step(i, carry):
        buf, R = carry
        origin = (axis_index_x - jnp.asarray(i, dtype=jnp.int32)) % px
        c_start = origin * jnp.asarray(c_chunk, dtype=jnp.int32)
        z = jnp.int32(0)
        psi_c_slice = lax.dynamic_slice(
            psi_c_Y, (z, c_start, z, z), (nk, c_chunk, nspinor, nu_local)
        )
        R = R + jnp.einsum("kcsN,bcvk->bvksN", jnp.conj(psi_c_slice), buf)
        buf = lax.ppermute(buf, axis_name="x", perm=perm)
        return buf, R

    _, R_total = lax.fori_loop(0, px, step, (X, R0))
    return R_total


def _ring_sum_valence_second(
    R: jax.Array,
    psi_v_X: jax.Array,
    v_chunk: int,
    py: int,
    mu_local: int,
) -> jax.Array:
    """Finish sum over valence: T(mu,nu,t,s,k) = sum_v psi_v(mu) R(v,k,s,nu)."""
    axis_index_y = jnp.asarray(lax.axis_index("y"), dtype=jnp.int32)
    nk, _, nspinor, _ = psi_v_X.shape
    nu_local = R.shape[4]

    T0 = jnp.zeros((R.shape[0], mu_local, nu_local, nspinor, nspinor, nk), dtype=R.dtype)
    perm = _ring_perm(py)

    def step(i, carry):
        buf, T = carry
        origin = (axis_index_y - jnp.asarray(i, dtype=jnp.int32)) % py
        v_start = origin * jnp.asarray(v_chunk, dtype=jnp.int32)
        z = jnp.int32(0)
        psi_v_slice = lax.dynamic_slice(
            psi_v_X, (z, v_start, z, z), (nk, v_chunk, nspinor, mu_local)
        )
        T = T + jnp.einsum("kvtM,bvksN->bMNtsk", psi_v_slice, buf)
        buf = lax.ppermute(buf, axis_name="y", perm=perm)
        return buf, T

    _, T_total = lax.fori_loop(0, py, step, (R, T0))
    return T_total


def apply_W_ring(
    X: jax.Array,
    psi_c_X: jax.Array,
    psi_v_Y: jax.Array,
    W_R: jax.Array,
    nkx: int,
    nky: int,
    nkz: int,
    px: int,
    py: int,
) -> jax.Array:
    nk = nkx * nky * nkz
    sqrt_nk = jnp.sqrt(jnp.asarray(nk, dtype=X.real.dtype))

    c_chunk = X.shape[1]
    v_chunk = X.shape[2]

    n_rmu_local_X = psi_c_X.shape[-1]
    n_rmu_local_Y = psi_v_Y.shape[-1]
    R = _ring_sum_valence(X, psi_v_Y, v_chunk, py, n_rmu_local_Y)
    T = _ring_sum_conduction(R, psi_c_X, c_chunk, px, n_rmu_local_X)

    nspinor = psi_c_X.shape[2]
    T_k = T.reshape(X.shape[0], n_rmu_local_X, n_rmu_local_Y, nspinor, nspinor, nkx, nky, nkz)
    T_R = jnp.fft.ifftn(T_k, axes=(5, 6, 7), norm="ortho")
    U_R = W_R[None, :, :, None, None, :, :, :] * T_R
    U_q = jnp.fft.fftn(U_R, axes=(5, 6, 7), norm="ortho")
    U = U_q.reshape(X.shape[0], n_rmu_local_X, n_rmu_local_Y, nspinor, nspinor, nk)

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
    nb_trial = X.shape[0]
    sqrt_nk = jnp.sqrt(jnp.asarray(nk, dtype=X.real.dtype))

    c_chunk = X.shape[1]
    v_chunk = X.shape[2]

    axis_index_x = jnp.asarray(lax.axis_index("x"), dtype=jnp.int32)
    axis_index_y = jnp.asarray(lax.axis_index("y"), dtype=jnp.int32)

    nk_local = psi_c_Y.shape[0]
    nspinor = psi_c_Y.shape[2]
    nu_local = psi_c_Y.shape[-1]
    mu_local = psi_c_X.shape[-1]

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

    _, A_local = lax.fori_loop(0, py, step_y, (X, A0))

    S0 = jnp.zeros((nb_trial, nu_local, nk_local), dtype=X.dtype)
    perm_x = _ring_perm(px)

    def step_x(i, carry):
        buf, S = carry
        S = S + jnp.sum(buf, axis=1)
        buf = lax.ppermute(buf, axis_name="x", perm=perm_x)
        return buf, S

    _, S_total = lax.fori_loop(0, px, step_x, (A_local, S0))

    S_total = S_total / sqrt_nk

    U_partial = jnp.einsum("MN,bNk->bMk", V_q0, S_total)
    U = lax.psum(U_partial, axis_name="y")

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
    W_R: jax.Array,
    V_q0: jax.Array,
    px: int,
    py: int,
) -> jax.Array:
    nk = nkx * nky * nkz
    if X.shape[1] % px != 0 or X.shape[2] % py != 0:
        raise ValueError("X band dimensions must be divisible by px/py for ring matvec")

    axis_index_x = jnp.asarray(lax.axis_index("x"), dtype=jnp.int32)
    axis_index_y = jnp.asarray(lax.axis_index("y"), dtype=jnp.int32)
    c_chunk = X.shape[1]
    v_chunk = X.shape[2]
    c_chunk_i = jnp.asarray(c_chunk, dtype=jnp.int32)
    v_chunk_i = jnp.asarray(v_chunk, dtype=jnp.int32)
    z = jnp.int32(0)
    eps_c_local = lax.dynamic_slice(eps_c, (z, axis_index_x * c_chunk_i), (nk, c_chunk))
    eps_v_local = lax.dynamic_slice(eps_v, (z, axis_index_y * v_chunk_i), (nk, v_chunk))
    delta_E = eps_c_local.T[None, :, None, :] - eps_v_local.T[None, None, :, :]
    D_term = delta_E * X
    V_term = apply_V_ring(X, psi_c_Y, psi_v_Y, psi_c_X, psi_v_X, V_q0, nk, px, py)
    W_term = apply_W_ring(X, psi_c_X, psi_v_Y, W_R, nkx, nky, nkz, px, py)

    return D_term + V_term - W_term


def build_bse_ring_matvec(
    mesh_xy: Mesh,
    nkx: int,
    nky: int,
    nkz: int,
    timed: bool = False,
    low_mem: bool = True,
    include_W: bool = True,
):
    px, py = mesh_xy.devices.shape
    sh = make_bse_shardings(mesh_xy)
    nk = nkx * nky * nkz

    def _encode_T(X, psi_c_X, psi_v_Y):
        c_chunk = X.shape[1]
        v_chunk = X.shape[2]
        n_rmu_local_X = psi_c_X.shape[-1]
        n_rmu_local_Y = psi_v_Y.shape[-1]
        R = _ring_sum_valence(X, psi_v_Y, v_chunk, py, n_rmu_local_Y)
        T = _ring_sum_conduction(R, psi_c_X, c_chunk, px, n_rmu_local_X)
        return T

    encode_T_ring = _shard_map_fn(
        _encode_T,
        mesh=mesh_xy,
        in_specs=(P(None, "x", "y", None), P(None, None, None, "x"), P(None, None, None, "y")),
        out_specs=P(None, "x", "y", None, None, None),
    )

    def _encode_T_gather(X, psi_c_X, psi_v_Y):
        X_full_v = lax.all_gather(X, "y", axis=2, tiled=True)
        R = jnp.einsum("kvsN,bcvk->bcksN", jnp.conj(psi_v_Y), X_full_v)
        R_full_c = lax.all_gather(R, "x", axis=1, tiled=True)
        T = jnp.einsum("kctM,bcksN->bMNtsk", psi_c_X, R_full_c)
        return T

    encode_T_gather = _shard_map_fn(
        _encode_T_gather,
        mesh=mesh_xy,
        in_specs=(P(None, "x", "y", None), P(None, None, None, "x"), P(None, None, None, "y")),
        out_specs=P(None, "x", "y", None, None, None),
    )

    def _apply_V_ring_only(X, psi_c_Y, psi_v_Y, psi_c_X, psi_v_X, V_q0):
        return apply_V_ring(X, psi_c_Y, psi_v_Y, psi_c_X, psi_v_X, V_q0, nk, px, py)

    apply_V_ring_only = _shard_map_fn(
        _apply_V_ring_only,
        mesh=mesh_xy,
        in_specs=(P(None, "x", "y", None), P(None, None, None, "y"), P(None, None, None, "y"),
                  P(None, None, None, "x"), P(None, None, None, "x"), P("x", "y")),
        out_specs=P(None, "x", "y", None),
    )

    # Custom-partitioned FFTs on the (kx, ky, kz) axes — those axes are
    # ``None``-sharded in T (sh.T) and W_R (sh.W), so the FFT can run
    # locally on every device.  Plain ``jnp.fft.ifftn`` / ``fftn`` on a
    # sharded tensor forces XLA to all-gather the entire array before
    # the FFT — see ``common.fft_helpers`` for the JAX bug this works
    # around.  In the BSE Lanczos loop those gathers cost ~5 s over
    # 200 matvecs on Si 4×4×4 (profile_sharded_v2/trace_summary.md).
    # T_k 8D spec: (b, μ, ν, ns, ns, kx, ky, kz) — same μ,ν shardings as
    # storage T (6D) but with last nk axis split into 3 replicated dims.
    _T_8d_spec = P(None, "x", "y", None, None, None, None, None)
    _T_local_ifftn = make_sharded_ifftn_3d(
        mesh_xy, _T_8d_spec, _T_8d_spec, axes=(5, 6, 7), norm='ortho')
    _T_local_fftn = make_sharded_fftn_3d(
        mesh_xy, _T_8d_spec, _T_8d_spec, axes=(5, 6, 7), norm='ortho')

    def _apply_W_from_T(T, psi_c_X, psi_v_Y, W_R):
        nspinor = psi_c_X.shape[2]
        nb_trial = T.shape[0]
        n_rmu_local_X = T.shape[1]
        n_rmu_local_Y = T.shape[2]
        sqrt_nk = jnp.sqrt(jnp.asarray(nk, dtype=T.real.dtype))

        T_k = T.reshape(nb_trial, n_rmu_local_X, n_rmu_local_Y, nspinor, nspinor, nkx, nky, nkz)
        T_R = _T_local_ifftn(T_k)
        U_R = W_R[None, :, :, None, None, :, :, :] * T_R
        U_q = _T_local_fftn(U_R)
        U = U_q.reshape(nb_trial, n_rmu_local_X, n_rmu_local_Y, nspinor, nspinor, nk)

        A = jnp.einsum("kctM,bMNtsk->bcNsk", jnp.conj(psi_c_X), U)
        WX = jnp.einsum("kvsN,bcNsk->bcvk", psi_v_Y, A)
        return WX / sqrt_nk

    apply_W_from_T = jax.jit(
        _apply_W_from_T,
        in_shardings=(sh.T, sh.psi_x, sh.psi_y, sh.W),
        out_shardings=sh.X,
        donate_argnums=(0,),  # T consumed once, no need to keep
    )

    def _apply_D_term(X, eps_c, eps_v):
        delta_E = eps_c.T[None, :, None, :] - eps_v.T[None, None, :, :]
        return delta_E * X

    apply_D_term = jax.jit(
        _apply_D_term,
        in_shardings=(sh.X, sh.eps, sh.eps),
        out_shardings=sh.X,
    )

    def _matvec_impl(X, psi_c_X, psi_c_Y, psi_v_X, psi_v_Y, eps_c, eps_v, W_R, V_q0):
        D_term = apply_D_term(X, eps_c, eps_v)
        V_term = apply_V_ring_only(X, psi_c_Y, psi_v_Y, psi_c_X, psi_v_X, V_q0)
        if not include_W:
            return D_term + V_term
        T = encode_T_ring(X, psi_c_X, psi_v_Y) if low_mem else encode_T_gather(X, psi_c_X, psi_v_Y)
        W_term = apply_W_from_T(T, psi_c_X, psi_v_Y, W_R)
        return D_term + V_term - W_term

    if timed:
        def _matvec_timed(X, psi_c_X, psi_c_Y, psi_v_X, psi_v_Y, eps_c, eps_v, W_R, V_q0):
            with timing.section("bse_jax.D_term"):
                D_term = apply_D_term(X, eps_c, eps_v)
                D_term.block_until_ready()
            with timing.section("bse_jax.V_term"):
                V_term = apply_V_ring_only(X, psi_c_Y, psi_v_Y, psi_c_X, psi_v_X, V_q0)
                V_term.block_until_ready()
            if not include_W:
                return D_term + V_term
            with timing.section("bse_jax.W_term"):
                T = encode_T_ring(X, psi_c_X, psi_v_Y) if low_mem else encode_T_gather(X, psi_c_X, psi_v_Y)
                W_term = apply_W_from_T(T, psi_c_X, psi_v_Y, W_R)
                W_term.block_until_ready()
            return D_term + V_term - W_term

        return _matvec_timed

    return jax.jit(
        _matvec_impl,
        in_shardings=(
            sh.X,
            sh.psi_x,
            sh.psi_y,
            sh.psi_x,
            sh.psi_y,
            sh.eps,
            sh.eps,
            sh.W,
            sh.V,
        ),
        out_shardings=sh.X,
    )


def build_bse_ring_matvec_full(
    mesh_xy: Mesh,
    nkx: int,
    nky: int,
    nkz: int,
    timed: bool = False,
    low_mem: bool = True,
    include_W: bool = True,
):
    """Build full (non-TDA) BSE matvec for S = [[A, B], [-B^H, -A^H]]."""
    px, py = mesh_xy.devices.shape
    sh = make_bse_shardings(mesh_xy)
    nk = nkx * nky * nkz

    def _encode_T(X, psi_c_X, psi_v_Y):
        c_chunk = X.shape[1]
        v_chunk = X.shape[2]
        n_rmu_local_X = psi_c_X.shape[-1]
        n_rmu_local_Y = psi_v_Y.shape[-1]
        R = _ring_sum_valence(X, psi_v_Y, v_chunk, py, n_rmu_local_Y)
        T = _ring_sum_conduction(R, psi_c_X, c_chunk, px, n_rmu_local_X)
        return T

    encode_T_ring = _shard_map_fn(
        _encode_T,
        mesh=mesh_xy,
        in_specs=(P(None, "x", "y", None), P(None, None, None, "x"), P(None, None, None, "y")),
        out_specs=P(None, "x", "y", None, None, None),
    )

    def _encode_T_gather(X, psi_c_X, psi_v_Y):
        X_full_v = lax.all_gather(X, "y", axis=2, tiled=True)
        R = jnp.einsum("kvsN,bcvk->bcksN", jnp.conj(psi_v_Y), X_full_v)
        R_full_c = lax.all_gather(R, "x", axis=1, tiled=True)
        T = jnp.einsum("kctM,bcksN->bMNtsk", psi_c_X, R_full_c)
        return T

    encode_T_gather = _shard_map_fn(
        _encode_T_gather,
        mesh=mesh_xy,
        in_specs=(P(None, "x", "y", None), P(None, None, None, "x"), P(None, None, None, "y")),
        out_specs=P(None, "x", "y", None, None, None),
    )

    def _encode_T_B(X, psi_c_Y, psi_v_X):
        c_chunk = X.shape[1]
        v_chunk = X.shape[2]
        n_rmu_local_X = psi_v_X.shape[-1]
        n_rmu_local_Y = psi_c_Y.shape[-1]
        R = _ring_sum_conduction_first(X, psi_c_Y, c_chunk, px, n_rmu_local_Y)
        T = _ring_sum_valence_second(R, psi_v_X, v_chunk, py, n_rmu_local_X)
        return T

    encode_T_ring_B = _shard_map_fn(
        _encode_T_B,
        mesh=mesh_xy,
        in_specs=(P(None, "x", "y", None), P(None, None, None, "y"), P(None, None, None, "x")),
        out_specs=P(None, "x", "y", None, None, None),
    )

    def _encode_T_B_gather(X, psi_c_Y, psi_v_X):
        X_full_c = lax.all_gather(X, "x", axis=1, tiled=True)
        R = jnp.einsum("kcsN,bcvk->bvksN", jnp.conj(psi_c_Y), X_full_c)
        R_full_v = lax.all_gather(R, "y", axis=1, tiled=True)
        T = jnp.einsum("kvtM,bvksN->bMNtsk", psi_v_X, R_full_v)
        return T

    encode_T_gather_B = _shard_map_fn(
        _encode_T_B_gather,
        mesh=mesh_xy,
        in_specs=(P(None, "x", "y", None), P(None, None, None, "y"), P(None, None, None, "x")),
        out_specs=P(None, "x", "y", None, None, None),
    )

    def _apply_V_ring_only(X, psi_c_Y, psi_v_Y, psi_c_X, psi_v_X, V_q0):
        return apply_V_ring(X, psi_c_Y, psi_v_Y, psi_c_X, psi_v_X, V_q0, nk, px, py)

    apply_V_ring_only = _shard_map_fn(
        _apply_V_ring_only,
        mesh=mesh_xy,
        in_specs=(P(None, "x", "y", None), P(None, None, None, "y"), P(None, None, None, "y"),
                  P(None, None, None, "x"), P(None, None, None, "x"), P("x", "y")),
        out_specs=P(None, "x", "y", None),
    )

    def _apply_V_ring_B(X, psi_c_Y, psi_v_Y, psi_c_X, psi_v_X, V_q0):
        return apply_V_ring(
            X,
            jnp.conj(psi_c_Y),
            jnp.conj(psi_v_Y),
            psi_c_X,
            psi_v_X,
            V_q0,
            nk,
            px,
            py,
        )

    apply_V_ring_B = _shard_map_fn(
        _apply_V_ring_B,
        mesh=mesh_xy,
        in_specs=(P(None, "x", "y", None), P(None, None, None, "y"), P(None, None, None, "y"),
                  P(None, None, None, "x"), P(None, None, None, "x"), P("x", "y")),
        out_specs=P(None, "x", "y", None),
    )

    # Custom-partitioned FFTs on the (kx, ky, kz) axes — those axes are
    # ``None``-sharded in T (sh.T) and W_R (sh.W), so the FFT can run
    # locally on every device.  Plain ``jnp.fft.ifftn`` / ``fftn`` on a
    # sharded tensor forces XLA to all-gather the entire array before
    # the FFT — see ``common.fft_helpers`` for the JAX bug this works
    # around.  In the BSE Lanczos loop those gathers cost ~5 s over
    # 200 matvecs on Si 4×4×4 (profile_sharded_v2/trace_summary.md).
    # T_k 8D spec: (b, μ, ν, ns, ns, kx, ky, kz) — same μ,ν shardings as
    # storage T (6D) but with last nk axis split into 3 replicated dims.
    _T_8d_spec = P(None, "x", "y", None, None, None, None, None)
    _T_local_ifftn = make_sharded_ifftn_3d(
        mesh_xy, _T_8d_spec, _T_8d_spec, axes=(5, 6, 7), norm='ortho')
    _T_local_fftn = make_sharded_fftn_3d(
        mesh_xy, _T_8d_spec, _T_8d_spec, axes=(5, 6, 7), norm='ortho')

    def _apply_W_from_T(T, psi_c_X, psi_v_Y, W_R):
        nspinor = psi_c_X.shape[2]
        nb_trial = T.shape[0]
        n_rmu_local_X = T.shape[1]
        n_rmu_local_Y = T.shape[2]
        sqrt_nk = jnp.sqrt(jnp.asarray(nk, dtype=T.real.dtype))

        T_k = T.reshape(nb_trial, n_rmu_local_X, n_rmu_local_Y, nspinor, nspinor, nkx, nky, nkz)
        T_R = _T_local_ifftn(T_k)
        U_R = W_R[None, :, :, None, None, :, :, :] * T_R
        U_q = _T_local_fftn(U_R)
        U = U_q.reshape(nb_trial, n_rmu_local_X, n_rmu_local_Y, nspinor, nspinor, nk)

        A = jnp.einsum("kctM,bMNtsk->bcNsk", jnp.conj(psi_c_X), U)
        WX = jnp.einsum("kvsN,bcNsk->bcvk", psi_v_Y, A)
        return WX / sqrt_nk

    apply_W_from_T = jax.jit(
        _apply_W_from_T,
        in_shardings=(sh.T, sh.psi_x, sh.psi_y, sh.W),
        out_shardings=sh.X,
        donate_argnums=(0,),  # T consumed once, no need to keep
    )

    def _apply_D_term(X, eps_c, eps_v):
        delta_E = eps_c.T[None, :, None, :] - eps_v.T[None, None, :, :]
        return delta_E * X

    apply_D_term = jax.jit(
        _apply_D_term,
        in_shardings=(sh.X, sh.eps, sh.eps),
        out_shardings=sh.X,
    )

    def _apply_A(X, psi_c_X, psi_c_Y, psi_v_X, psi_v_Y, eps_c, eps_v, W_R, V_q0):
        D_term = apply_D_term(X, eps_c, eps_v)
        V_term = apply_V_ring_only(X, psi_c_Y, psi_v_Y, psi_c_X, psi_v_X, V_q0)
        if not include_W:
            return D_term + V_term
        T = encode_T_ring(X, psi_c_X, psi_v_Y) if low_mem else encode_T_gather(X, psi_c_X, psi_v_Y)
        W_term = apply_W_from_T(T, psi_c_X, psi_v_Y, W_R)
        return D_term + V_term - W_term

    def _apply_B(X, psi_c_X, psi_c_Y, psi_v_X, psi_v_Y, W_R, V_q0):
        V_term = apply_V_ring_B(X, psi_c_Y, psi_v_Y, psi_c_X, psi_v_X, V_q0)
        if not include_W:
            return V_term
        T = encode_T_ring_B(X, psi_c_Y, psi_v_X) if low_mem else encode_T_gather_B(X, psi_c_Y, psi_v_X)
        W_term = apply_W_from_T(T, psi_c_X, psi_v_Y, W_R)
        return V_term - W_term

    def _matvec_impl(X_full, psi_c_X, psi_c_Y, psi_v_X, psi_v_Y, eps_c, eps_v, W_R, V_q0):
        X = X_full[0]
        Y = X_full[1]
        AX = _apply_A(X, psi_c_X, psi_c_Y, psi_v_X, psi_v_Y, eps_c, eps_v, W_R, V_q0)
        BY = _apply_B(Y, psi_c_X, psi_c_Y, psi_v_X, psi_v_Y, W_R, V_q0)
        X_out = AX + BY

        # A and B are Hermitian in this formulation; reuse A(Y) and B(X).
        AY = _apply_A(Y, psi_c_X, psi_c_Y, psi_v_X, psi_v_Y, eps_c, eps_v, W_R, V_q0)
        B_dag_X = _apply_B(X, psi_c_X, psi_c_Y, psi_v_X, psi_v_Y, W_R, V_q0)
        Y_out = -B_dag_X - AY
        return jnp.stack([X_out, Y_out], axis=0)

    if timed:
        def _matvec_timed(X_full, psi_c_X, psi_c_Y, psi_v_X, psi_v_Y, eps_c, eps_v, W_R, V_q0):
            X = X_full[0]
            Y = X_full[1]
            with timing.section("bse_jax.D_term"):
                D_term = apply_D_term(X, eps_c, eps_v)
                D_term.block_until_ready()
            with timing.section("bse_jax.V_term"):
                V_term = apply_V_ring_only(X, psi_c_Y, psi_v_Y, psi_c_X, psi_v_X, V_q0)
                V_term.block_until_ready()
            if not include_W:
                AX = D_term + V_term
            else:
                with timing.section("bse_jax.W_term"):
                    T = encode_T_ring(X, psi_c_X, psi_v_Y) if low_mem else encode_T_gather(X, psi_c_X, psi_v_Y)
                    W_term = apply_W_from_T(T, psi_c_X, psi_v_Y, W_R)
                    W_term.block_until_ready()
                AX = D_term + V_term - W_term
            with timing.section("bse_jax.B_term"):
                BY = _apply_B(Y, psi_c_X, psi_c_Y, psi_v_X, psi_v_Y, W_R, V_q0)
                BY.block_until_ready()
            X_out = AX + BY

            AY = _apply_A(Y, psi_c_X, psi_c_Y, psi_v_X, psi_v_Y, eps_c, eps_v, W_R, V_q0)
            B_dag_X = _apply_B(X, psi_c_X, psi_c_Y, psi_v_X, psi_v_Y, W_R, V_q0)
            Y_out = -B_dag_X - AY
            return jnp.stack([X_out, Y_out], axis=0)

        return _matvec_timed

    return jax.jit(
        _matvec_impl,
        in_shardings=(
            sh.X_full,
            sh.psi_x,
            sh.psi_y,
            sh.psi_x,
            sh.psi_y,
            sh.eps,
            sh.eps,
            sh.W,
            sh.V,
        ),
        out_shardings=sh.X_full,
    )


def build_realspace_random_transition_generator(
    mesh_xy: Mesh,
    nkx: int,
    nky: int,
    nkz: int,
    n_cond_pad: int,
    n_val_pad: int,
):
    """Build a sharded map: r(nu,k) -> V_q0 @ r -> X(c,v,k) (local c/v chunks).

    Assumes r is column-sharded on the Y axis and V_q0 is tiled (x,y).
    The resulting X is sharded like the BSE matvec (c on x, v on y).
    """
    px, py = mesh_xy.devices.shape
    sh = make_bse_shardings(mesh_xy)
    nk = nkx * nky * nkz

    if n_cond_pad % px != 0 or n_val_pad % py != 0:
        raise ValueError("n_cond_pad and n_val_pad must be divisible by px/py")

    c_chunk = n_cond_pad // px
    v_chunk = n_val_pad // py

    def _map(r, psi_c_X, psi_v_X, V_q0):
        # r: (batch, nu_local, nk), column-sharded on y.
        U_partial = jnp.einsum("MN,bNk->bMk", V_q0, r)
        U = lax.psum(U_partial, axis_name="y")  # (batch, mu_local, nk), sharded on x

        axis_index_x = jnp.asarray(lax.axis_index("x"), dtype=jnp.int32)
        axis_index_y = jnp.asarray(lax.axis_index("y"), dtype=jnp.int32)
        c_start = axis_index_x * jnp.asarray(c_chunk, dtype=jnp.int32)
        v_start = axis_index_y * jnp.asarray(v_chunk, dtype=jnp.int32)
        z = jnp.int32(0)

        nk_local, _, nspinor, mu_local = psi_c_X.shape
        psi_c_slice = lax.dynamic_slice(
            psi_c_X, (z, c_start, z, z), (nk_local, c_chunk, nspinor, mu_local)
        )
        psi_v_slice = lax.dynamic_slice(
            psi_v_X, (z, v_start, z, z), (nk_local, v_chunk, nspinor, mu_local)
        )

        M_X = jnp.einsum("kcsm,kvsm->kcvm", jnp.conj(psi_c_slice), psi_v_slice)
        X_local = jnp.einsum("kcvM,bMk->bcvk", jnp.conj(M_X), U)

        sqrt_nk = jnp.sqrt(jnp.asarray(nk, dtype=X_local.real.dtype))
        return X_local / sqrt_nk

    return _shard_map_fn(
        _map,
        mesh=mesh_xy,
        in_specs=(P(None, "y", None), P(None, None, None, "x"), P(None, None, None, "x"), P("x", "y")),
        out_specs=P(None, "x", "y", None),
    )


def build_density_snapshot_operator(
    mesh_xy: Mesh,
    nkx: int,
    nky: int,
    nkz: int,
):
    """Build a sharded map: s(c,v,k) -> d(mu) = V_q0 @ sum_k R s.

    Returns density-space vectors in r_mu (summed over k) sharded on x.
    """
    px, py = mesh_xy.devices.shape
    sh = make_bse_shardings(mesh_xy)
    nk = nkx * nky * nkz

    def _map(s, psi_c_Y, psi_v_Y, V_q0):
        nb_trial = s.shape[0]
        sqrt_nk = jnp.sqrt(jnp.asarray(nk, dtype=s.real.dtype))

        c_chunk = s.shape[1]
        v_chunk = s.shape[2]

        axis_index_x = jnp.asarray(lax.axis_index("x"), dtype=jnp.int32)
        axis_index_y = jnp.asarray(lax.axis_index("y"), dtype=jnp.int32)

        nk_local = psi_c_Y.shape[0]
        nspinor = psi_c_Y.shape[2]
        nu_local = psi_c_Y.shape[-1]

        c_start_local = axis_index_x * jnp.asarray(c_chunk, dtype=jnp.int32)
        z = jnp.int32(0)
        psi_c_slice = lax.dynamic_slice(
            psi_c_Y, (z, c_start_local, z, z), (nk_local, c_chunk, nspinor, nu_local)
        )

        A0 = jnp.zeros((nb_trial, c_chunk, nu_local, nk_local), dtype=s.dtype)
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

        _, A_local = lax.fori_loop(0, py, step_y, (s, A0))

        S0 = jnp.zeros((nb_trial, nu_local, nk_local), dtype=s.dtype)
        perm_x = _ring_perm(px)

        def step_x(i, carry):
            buf, S = carry
            S = S + jnp.sum(buf, axis=1)
            buf = lax.ppermute(buf, axis_name="x", perm=perm_x)
            return buf, S

        _, S_total = lax.fori_loop(0, px, step_x, (A_local, S0))

        S_total = S_total / sqrt_nk

        U_partial = jnp.einsum("MN,bNk->bMk", V_q0, S_total)
        U = lax.psum(U_partial, axis_name="y")  # (batch, mu_local, nk)

        d_mu = jnp.sum(U, axis=-1)
        return d_mu

    return _shard_map_fn(
        _map,
        mesh=mesh_xy,
        in_specs=(P(None, "x", "y", None), P(None, None, None, "y"), P(None, None, None, "y"), P("x", "y")),
        out_specs=P(None, "x"),
    )


def build_density_drive_operators(mesh_xy, nkx, nky, nkz, n_cond_pad, n_val_pad):
    """Return callable(s) to build density-driven RHS blocks in transition space.

    Transition vertex: d(t,mu) = sum_s conj(psi_c(t,mu)) * psi_v(t,mu)
    For density-space source g(mu), u = V g. Need f = d^dagger u, fbar = d^T u.
    For complex Bloch spinors, do not assume fbar == conj(f).
    """
    gen = build_realspace_random_transition_generator(mesh_xy, nkx, nky, nkz, n_cond_pad, n_val_pad)

    def f_op(r, psi_c_X, psi_v_X, V_q0):
        return gen(r, psi_c_X, psi_v_X, V_q0)

    def fbar_op(r, psi_c_X, psi_v_X, V_q0):
        return gen(r, jnp.conj(psi_c_X), jnp.conj(psi_v_X), V_q0)

    return f_op, fbar_op


def build_density_readout_operator_full(mesh_xy, nkx, nky, nkz):
    """Return a callable to compute w = V (d X + d* Y) from X_full=[X,Y]."""
    snapshot = build_density_snapshot_operator(mesh_xy, nkx, nky, nkz)

    def _readout(X_full, psi_c_Y, psi_v_Y, V_q0):
        X = X_full[0]
        Y = X_full[1]
        w_x = snapshot(X, psi_c_Y, psi_v_Y, V_q0)
        w_y = snapshot(Y, jnp.conj(psi_c_Y), jnp.conj(psi_v_Y), V_q0)
        return w_x + w_y

    return _readout


def ring_matvec_smoke_test(px: int = 2, py: int = 2) -> None:
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
        W_R = jnp.fft.ifftn(W_q, axes=(2, 3, 4), norm="ortho")
        HX = matvec(X, psi_c_X, psi_c_Y, psi_v_X, psi_v_Y, eps_c, eps_v, W_R, V_q0)
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

    payload = _load_ring_subset(restart_file, n_val, n_cond, px, py,
                                input_file=input_file)
    psi_c = payload["psi_c"]
    psi_v = payload["psi_v"]
    eps_c = payload["eps_c"]
    eps_v = payload["eps_v"]
    W_q = payload["W_q"]
    V_q0 = payload["V_q0"]
    X = payload["X"]
    nkx = payload["nkx"]
    nky = payload["nky"]
    nkz = payload["nkz"]
    nk = payload["nk"]
    n_rmu_pad = payload["n_rmu_pad"]

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
        W_R = jnp.fft.ifftn(W_q, axes=(2, 3, 4), norm="ortho")

        matvec = build_bse_ring_matvec(mesh, nkx, nky, nkz)
        HX_ring = matvec(X, psi_c_X, psi_c_Y, psi_v_X, psi_v_Y, eps_c, eps_v, W_R, V_q0)
        HX_ring.block_until_ready()

    HX_ring_host = jax.device_get(HX_ring)
    diff = jnp.linalg.norm(HX_ring_host - HX_ref) / jnp.maximum(jnp.linalg.norm(HX_ref), 1e-12)
    print(f"Relative error ||HX_ring - HX_ref||/||HX_ref||: {float(diff):.3e}")

    if component_check:
        sqrt_nk = jnp.sqrt(jnp.asarray(nk, dtype=jnp.float64))
        M = compute_pair_amplitude(psi_c, psi_v)
        D_ref = apply_D(X, eps_c, eps_v)
        S_V = jnp.einsum("kcvN,bcvk->bNk", M, X) / sqrt_nk
        U_V = jnp.einsum("MN,bNk->bMk", V_q0, S_V)
        V_ref = jnp.einsum("kcvM,bMk->bcvk", jnp.conj(M), U_V) / sqrt_nk

        R = jnp.einsum("kvsN,bcvk->bcksN", jnp.conj(psi_v), X)
        T = jnp.einsum("kctM,bcksN->bMNtsk", psi_c, R)
        T_k = T.reshape(X.shape[0], n_rmu_pad, n_rmu_pad, psi_c.shape[2], psi_c.shape[2], nkx, nky, nkz)
        T_R = jnp.fft.ifftn(T_k, axes=(5, 6, 7), norm="ortho")
        W_R = jnp.fft.ifftn(W_q, axes=(2, 3, 4), norm="ortho")
        U_R = W_R[None, :, :, None, None, :, :, :] * T_R
        U_q = jnp.fft.fftn(U_R, axes=(5, 6, 7), norm="ortho")
        U = U_q.reshape(X.shape[0], n_rmu_pad, n_rmu_pad, psi_c.shape[2], psi_c.shape[2], nk)
        A = jnp.einsum("kctM,bMNtsk->bcNsk", jnp.conj(psi_c), U)
        W_ref = jnp.einsum("kvsN,bcNsk->bcvk", psi_v, A) / sqrt_nk

        comp_matvec = build_bse_ring_matvec(mesh, nkx, nky, nkz)
        with mesh:
            D_ring = apply_D(X, eps_c, eps_v)
            W_R = jnp.fft.ifftn(W_q, axes=(2, 3, 4), norm="ortho")
            V_ring = comp_matvec(
                X, psi_c_X, psi_c_Y, psi_v_X, psi_v_Y, eps_c * 0.0, eps_v * 0.0, W_R * 0.0, V_q0
            )
            W_ring = comp_matvec(
                X, psi_c_X, psi_c_Y, psi_v_X, psi_v_Y, eps_c * 0.0, eps_v * 0.0, W_R, V_q0 * 0.0
            )
            D_ring.block_until_ready()
            V_ring.block_until_ready()
            W_ring.block_until_ready()

        def _rel_err(a, b):
            return float(jnp.linalg.norm(a - b) / jnp.maximum(jnp.linalg.norm(b), 1e-12))

        print(f"Component error D: { _rel_err(D_ring, D_ref):.3e}")
        print(f"Component error V: { _rel_err(V_ring, V_ref):.3e}")
        print(f"Component error W: { _rel_err(-W_ring, W_ref):.3e}")


