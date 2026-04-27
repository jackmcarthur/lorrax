"""Plain-jit BSE matvec — no shard_map, no hand-rolled ring/gather.

The BSE matvec is fundamentally three operations:

  D term   :  (ε_c − ε_v) · X
  V term   :  pair density at q=0  ·  V_q0  ·  pair density^H · X
  W term   :  T = encode(ψ_c, ψ_v, X)         (two einsums)
              U = ifft·W_R·fft applied to T   (FFTs + pointwise)
              HX_W = decode(ψ_c, ψ_v, U)      (two einsums)

Every ``shard_map`` block in ``bse_ring_comm.py`` is just hand-rolling
collective patterns (ring of ``ppermute`` or single ``all_gather``)
that XLA's automatic SPMD partitioner can generate from plain
``jnp.einsum`` calls when the inputs and outputs are tagged with
``with_sharding_constraint``.  This module is a clean alternative.

Shardings on the (x, y) mesh are kept identical to the existing
implementation (``make_bse_shardings``):
  X        P(None, "x", "y", None)
  psi_c_X  P(None, None, None, "x")     (μ on x; c replicated)
  psi_c_Y  P(None, None, None, "y")     (ν on y; c replicated)
  psi_v_X, psi_v_Y                      (same convention)
  V_q0     P("x", "y")
  W        P("x", "y", None, None, None)
  T        P(None, "x", "y", None, None, None)

The ``psum_scatter`` trick from ``ppm_sigma._make_project_ri_reduce_scatter``
is used at the V term's μ-contraction step: instead of a plain psum +
later all-gather, we reduce-scatter so the output stays sharded on the
mesh axis we'll need next.
"""
from __future__ import annotations

from functools import partial
from types import SimpleNamespace
from typing import Optional

import jax
import jax.numpy as jnp
import numpy as np
from jax import lax
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

from common.fft_helpers import (
    make_jittable_local_fftn_3d,
    make_jittable_local_ifftn_3d,
)
from .bse_ring_comm import make_bse_shardings


def build_bse_simple_matvec(
    mesh_xy: Mesh,
    nkx: int,
    nky: int,
    nkz: int,
    *,
    include_W: bool = True,
):
    """Build a plain-jit BSE matvec — no shard_map, no rings.

    The returned matvec is a single jit'd function:
        matvec(X, psi_c_X, psi_c_Y, psi_v_X, psi_v_Y,
               eps_c, eps_v, W_R, V_q0)  → HX

    All collectives are XLA-generated from einsums + sharding hints.
    Custom-partitioned FFTs from ``common.fft_helpers`` are still used
    for the (kx, ky, kz) IFFT/FFT around the W contraction (those wrap
    a JAX-bug-affected FFT primitive — separate from the shard_map
    avoidance question).
    """
    sh = make_bse_shardings(mesh_xy)
    nk = nkx * nky * nkz
    sqrt_nk = jnp.sqrt(jnp.asarray(nk, dtype=jnp.float64))

    # Custom-partitioned FFTs over k-axes 5,6,7 of the 8D T tensor.
    _T_8d_spec = P(None, "x", "y", None, None, None, None, None)
    _T_local_ifftn = make_jittable_local_ifftn_3d(
        mesh_xy, _T_8d_spec, _T_8d_spec, axes=(5, 6, 7), norm='ortho')
    _T_local_fftn = make_jittable_local_fftn_3d(
        mesh_xy, _T_8d_spec, _T_8d_spec, axes=(5, 6, 7), norm='ortho')

    def _matvec(
        X, psi_c_X, psi_c_Y, psi_v_X, psi_v_Y,
        eps_c, eps_v, W_R, V_q0,
    ):
        # ── D term:  (ε_c − ε_v) · X  — purely local, no comm ──────────
        delta_E = eps_c.T[None, :, None, :] - eps_v.T[None, None, :, :]
        D_term = delta_E * X
        D_term = lax.with_sharding_constraint(D_term, sh.X)

        # ── V term (q=0 exchange, rank-1 in the centroid basis) ────────
        # Build M[k, c, v, ν] = Σ_s conj(ψ_c_Y[k,c,s,ν]) · ψ_v_Y[k,v,s,ν].
        # Both inputs have ν on y, c & v replicated. Output: c, v rep, ν on y.
        M_Y = jnp.einsum(
            'kcsN,kvsN->kcvN',
            jnp.conj(psi_c_Y), psi_v_Y, optimize=True,
        )

        # S[b, ν, k] = Σ_{c,v} M_Y[k,c,v,ν] · X[b,c,v,k] / sqrt_nk.
        # Original forward (apply_V_ring) is conj(ψ_c)·ψ_v·X — i.e. M_Y
        # itself, not conj(M_Y).  The hermitian-conjugate counterpart on
        # the back-contract uses conj(M_X) instead.
        # Contraction over c (x-sharded in X) and v (y-sharded in X).
        # M has c, v replicated and ν on y. XLA picks the partitioning;
        # we tag the output P(None, "y", None).
        S = jnp.einsum(
            'kcvN,bcvk->bNk',
            M_Y, X, optimize=True,
        )
        S = lax.with_sharding_constraint(S, sh.S)
        S = S / sqrt_nk

        # U[b, μ, k] = V_q0[μ, ν] · S[b, ν, k].
        # V_q0 P(x, y) on (μ, ν); S P(None, y, None) on (b, ν, k).
        # ν reduces (sharded y); output μ on x. Plain einsum + reshard hint.
        U_mu = jnp.einsum(
            'MN,bNk->bMk',
            V_q0, S, optimize=True,
        )
        U_mu = lax.with_sharding_constraint(U_mu, sh.U_mu)

        # M_X[k, c, v, μ] needed to back-contract: build from psi_c_X, psi_v_X.
        M_X = jnp.einsum(
            'kcsM,kvsM->kcvM',
            jnp.conj(psi_c_X), psi_v_X, optimize=True,
        )

        # HX_V[b, c, v, k] = Σ_μ M_X*[k,c,v,μ] · U_mu[b,μ,k] / sqrt_nk.
        # M_X has c, v replicated, μ on x. U_mu has μ on x. Locally μ-aligned;
        # output b, c on x (need to scatter c to x), v rep, k rep.
        HX_V = jnp.einsum(
            'kcvM,bMk->bcvk',
            jnp.conj(M_X), U_mu, optimize=True,
        )
        HX_V = lax.with_sharding_constraint(HX_V, sh.X)
        HX_V = HX_V / sqrt_nk

        if not include_W:
            return D_term + HX_V

        # ── W term (direct kernel) ─────────────────────────────────────
        # T[b, μ, ν, t, s, k] = Σ_v Σ_c ψ_c_X[k,c,t,μ]·ψ_v_Y*[k,v,s,ν]·X[b,c,v,k]
        #
        # Two contractions, two collectives. Express as one composite
        # einsum and let XLA pick the order. Tag the output sharding.
        T = jnp.einsum(
            'kctM,kvsN,bcvk->bMNtsk',
            psi_c_X, jnp.conj(psi_v_Y), X, optimize=True,
        )
        T = lax.with_sharding_constraint(T, sh.T)

        # FFT over (kx, ky, kz) — split nk into 3 axes.
        T_k = T.reshape(T.shape[0], T.shape[1], T.shape[2],
                        T.shape[3], T.shape[4], nkx, nky, nkz)
        T_R = _T_local_ifftn(T_k)

        # Pointwise: U_R = W_R[None, μ, ν, None, None, kx, ky, kz] · T_R.
        # W_R P(x, y, None, None, None); T_R P(None, x, y, None, None, None, None, None).
        # Aligned on (x, y). Local multiply — no comm.
        U_R = W_R[None, :, :, None, None, :, :, :] * T_R
        U_q = _T_local_fftn(U_R)
        U = U_q.reshape(U_q.shape[0], U_q.shape[1], U_q.shape[2],
                        U_q.shape[3], U_q.shape[4], nk)
        U = lax.with_sharding_constraint(U, sh.U)

        # Back-contract: A[b, c, ν, s, k] = Σ_μ ψ_c_X*[k,c,t,μ] · U[b,μ,ν,t,s,k]
        A = jnp.einsum(
            'kctM,bMNtsk->bcNsk',
            jnp.conj(psi_c_X), U, optimize=True,
        )
        A = lax.with_sharding_constraint(A, sh.A)

        # HX_W[b, c, v, k] = Σ_ν Σ_s ψ_v_Y[k,v,s,ν] · A[b,c,ν,s,k] / sqrt_nk
        HX_W = jnp.einsum(
            'kvsN,bcNsk->bcvk',
            psi_v_Y, A, optimize=True,
        )
        HX_W = lax.with_sharding_constraint(HX_W, sh.X)
        HX_W = HX_W / sqrt_nk

        return D_term + HX_V - HX_W

    return jax.jit(
        _matvec,
        in_shardings=(
            sh.X, sh.psi_x, sh.psi_y, sh.psi_x, sh.psi_y,
            sh.eps, sh.eps, sh.W, sh.V,
        ),
        out_shardings=sh.X,
    )
