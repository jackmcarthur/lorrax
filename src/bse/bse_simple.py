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

import jax
import jax.numpy as jnp
from jax import lax
from jax.sharding import Mesh, PartitionSpec as P

from common.fft_helpers import make_kconv_kminor
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
               eps_c, eps_v, W_R, V_q0, M_X, M_Y)  → HX

    ``M_X`` (μ on x) / ``M_Y`` (ν on y) are the hoisted exchange pair amplitudes
    (audit P3), precomputed once per solve; ``psi_c_Y``/``psi_v_X`` are retained
    only for a uniform signature with the other matvecs (the W-term reads
    ``psi_c_X``/``psi_v_Y``; the V-term reads the hoisted M's).

    All collectives are XLA-generated from einsums + sharding hints.
    The k-convolution around the W contraction is ONE call of the
    k-convolution router's k-minor door (``make_kconv_kminor``: nvidia-mathdx
    on CUDA, the plan route on cpu).
    """
    sh = make_bse_shardings(mesh_xy)
    nk = nkx * nky * nkz
    sqrt_nk = jnp.sqrt(jnp.asarray(nk, dtype=jnp.float64))

    _conv = make_kconv_kminor(mesh_xy, (nkx, nky, nkz),
                              P(None, "x", "y", None, None, None), P("x", "y", None),
                              norm='ortho')

    def _matvec(
        X, psi_c_X, psi_c_Y, psi_v_X, psi_v_Y,
        eps_c, eps_v, W_R, V_q0, M_X, M_Y,
    ):
        # M_X (μ on x) / M_Y (ν on y): hoisted exchange pair amplitudes, precomputed
        # once per solve (audit P3). psi_c_Y / psi_v_X are unused here — kept for a
        # uniform matvec signature; psi_c_X / psi_v_Y still feed the W-term.
        # ── D term:  (ε_c − ε_v) · X  — purely local, no comm ──────────
        delta_E = eps_c.T[None, :, None, :] - eps_v.T[None, None, :, :]
        D_term = delta_E * X
        D_term = lax.with_sharding_constraint(D_term, sh.X)

        # ── V term (q=0 exchange, rank-1 in the centroid basis) ────────
        # M_Y[k, c, v, ν] = Σ_s conj(ψ_c_Y) · ψ_v_Y (ν on y, c & v replicated), hoisted.

        # Exchange (q=0) is DENSE in (k,k') — k is SUMMED in the encode, one
        # V_q0 solve, then broadcast back at every k in the decode (VERDICT.md).
        # S[b, ν] = Σ_{k,c,v} conj(M_Y[k,c,v,ν]) · X[b,c,v,k] / sqrt_nk.
        # The transition density <0|ρ̂|Ψ> = Σ A_cvk ψ_ck ψ*_vk puts the
        # CONJUGATED vertex on the forward (encode) leg and the bare vertex on
        # the back-contract: K^x = M V M†.  The reverse assignment builds
        # conj(M) V M^T = conj(K^x), which no symmetry operator can commute
        # with alongside the (correct, untouched) W term.
        # Contraction over k (replicated), c (x-sharded in X), v (y-sharded).
        # M has c, v replicated and ν on y. XLA picks the partitioning;
        # we tag the output P(None, "y").
        S = jnp.einsum(
            'kcvN,bcvk->bN',
            jnp.conj(M_Y), X, optimize=True,
        )
        S = lax.with_sharding_constraint(S, sh.S_k0)
        S = S / sqrt_nk

        # U[b, μ] = V_q0[μ, ν] · S[b, ν].
        # V_q0 P(x, y) on (μ, ν); S P(None, y) on (b, ν).
        # ν reduces (sharded y); output μ on x. Plain einsum + reshard hint.
        U_mu = jnp.einsum(
            'MN,bN->bM',
            V_q0, S, optimize=True,
        )
        U_mu = lax.with_sharding_constraint(U_mu, sh.d_mu)

        # M_X[k, c, v, μ] (μ on x) is the hoisted back-contract vertex — BARE,
        # the conjugate having been spent on the forward leg above.
        # HX_V[b, c, v, k] = Σ_μ M_X[k,c,v,μ] · U_mu[b,μ] / sqrt_nk (broadcast k).
        # M_X has c, v replicated, μ on x. U_mu has μ on x. Locally μ-aligned;
        # output b, c on x (need to scatter c to x), v rep, k rep.
        HX_V = jnp.einsum(
            'kcvM,bM->bcvk',
            M_X, U_mu, optimize=True,
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

        # U = fftn_k(ifftn_k(T) · W_R), both 'ortho', one fused call; the
        # (μ, ν) tiles of T and W_R share the (x, y) mesh axes, so it is local.
        U = _conv(T, W_R.reshape(W_R.shape[0], W_R.shape[1], nk))
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
            sh.eps, sh.eps, sh.W, sh.V, sh.psi_x, sh.psi_y,
        ),
        out_shardings=sh.X,
    )
