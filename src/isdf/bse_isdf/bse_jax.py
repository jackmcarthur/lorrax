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

import numpy as np
import jax
import jax.numpy as jnp
from jax.sharding import Mesh, PartitionSpec as P, NamedSharding
from jax import lax

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
    # Energy difference: (nk, nc, nv) -> broadcast with X
    # eps_c: (nk, nc) -> (1, nc, 1, nk)
    # eps_v: (nk, nv) -> (1, 1, nv, nk)
    delta_E = eps_c.T[None, :, None, :] - eps_v.T[None, None, :, :]  # (1, nc, nv, nk)
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
    delta_E = eps_c.T[None, :, None, :] - eps_v.T[None, None, :, :]
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


if __name__ == "__main__":
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

