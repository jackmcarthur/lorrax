"""Batched trial-stack BSE matvec — one T-tensor alive regardless of n_trials.

``build_bse_stack_matvec`` returns a jitted

    matvec(X[n_trials, c, v, k], psi_c_X, psi_c_Y, psi_v_X, psi_v_Y,
           eps_c, eps_v, W_R, V_q0, M)  ->  out[n_trials, c, v, k]

for the TDA BSE (or RPA) Hamiltonian ``H = D + V - W`` (``D + V`` for RPA).

The exchange pair amplitude ``M[k,c,v,μ] = Σ_s conj(ψ_c) ψ_v`` is a pure function
of ψ (``compute_pair_amplitude``), so it is HOISTED to a matvec input
(precomputed ONCE per solve at load time, ``bse_io``) rather than rebuilt inside
every iteration — the matvec is a per-iteration black-box jit whose ψ args XLA
cannot hoist across calls (audit P3, ``reports/bse_refactor_map_2026-07-15/archive/
matvec_efficiency_audit``).  ONE copy, in the TRANSITION layout ``sh.M``: (c, v)
tiled exactly like X (c on x, v on y) and μ whole, so it is 1/P per rank and the
exchange encode and decode are both local contractions against the rank's own X
tile; the only collectives are two psums of a (n_trials, μ) vector
(``_exchange_U``).  It replaced two copies sharded on μ alone (``M_X`` μ on x,
``M_Y`` ν on y), 2·M/√P per rank.  ``psi_c_Y``/``psi_v_X`` are retained in the
signature for a uniform calling convention with the ring matvecs (they feed
only the coupling block's encode here).

Why a stack matvec.  A matvec that carries the trial axis ``b`` on the
direct-term tensor ``T[b, μ, ν, t, s, k]`` holds ``n_trials · μ_loc · ν_loc ·
ns² · nk`` complex128 per device, linear in ``n_trials``.  Here the W-term body
is a ``lax.scan`` over the trial axis, so XLA reuses the body's scratch across
iterations and exactly ONE ``T``-family is alive regardless of ``n_trials``.  A
Python-unrolled loop or a ``fori_loop`` over trials inside ``jit`` would keep
``n_trials`` live ``T`` slots; the scan does not.

Inside the trial scan the W term is a sum over (μ, ν) blocks of the rank's
``W_R`` tile (``_blocked_w_term``, planned from the shapes by
``plan_w_blocks``): what one trial stores is one block's U, never the tile's,
and every outer-load handler call stays below 2^31 elements.  One block is the
unblocked program.  The design is ``docs/architecture/bse.md#the-matvec``.

The ``lax.scan`` bounds ``T``, not the ``shard_map``: the GSPMD twin below runs
the same scan with no ``shard_map`` and holds 450.01 MiB at ``n_trials=1``
against 450.10 MiB at ``n_trials=8`` (measured 2026-08-08).  The manual
``shard_map`` is kept for two reasons:

  1. BACKEND PORTABILITY of the decode collectives.  On a CPU mesh the SPMD
     partitioner emits ``all-reduce`` where the manual body issues
     ``psum_scatter`` — 2x the wire bytes on BOTH decode legs, with no flag in
     this build that fixes it.
  2. A GUARANTEE RATHER THAN A COINCIDENCE.  On GPU the partitioner reproduces
     the manual plan exactly (same 6 collectives, same bytes), but that is a
     property of this XLA build, not a contract it owes us.  The manual
     spelling cannot silently regress.

Exchange (V) is the B1 dense form (VERDICT.md): DENSE in (k,k'), encode k-SUMMED
into a k-free ζ-space density, decode broadcast at every k.  ``S,U`` are k-free
(tiny, ``n_trials × ν``) so the V term stays outside the scan, batched.

Shardings come from ``make_bse_shardings``; ``n_trials`` occupies the leading
axis of ``sh.X = P(None,'x','y',None)``.

The W-tile seam is the single line ``U = fft_k(W_R * ifft_k(T))``: ``W_R`` is a
shape-stable ``(μ_pad, ν_pad, nkx, nky, nkz)`` argument built ONCE outside the
matvec, so W(ω) / ladder buildouts pass a different ``W_R`` with no change to
encode/decode/scan.

Consumers: ``bse_lanczos.solve_bse_sharded`` (block Lanczos and Davidson),
``bse_feast`` (TDA GMRES contour solves, ``_rayleigh_ritz``, the spectral
bounds), ``bse_kpm`` and the pseudopole and Haydock paths.  The non-TDA
coupling block's encode is :func:`build_bse_stack_pair_matvec` below.
``bse_ring_comm.build_bse_ring_matvec_full`` stays as the ``_materialize_A_B``
oracle and the equality gate's twin.  ``bse_preconditioner.exchange_spin_weight``
owns the scalar-singlet exchange weight for both.

THE COUPLING BLOCK, AND THE FUSION THAT PAYS FOR IT
---------------------------------------------------
``build_bse_stack_pair_matvec`` returns the SDY real-linear pair applier

    pair(X, s, ...)  =  A·X  +  s·B·conj(X)

with ``s = +1`` giving Shao-da Jornada-Yang's ``F(x) = Ax + Bx̄`` and ``s = -1``
giving their ``G(v) = Av - Bv̄`` (Algorithm 4, arXiv:1611.02348; the derivation
this implements is ``NONTDA_MATRIXFREE_DERIVATION.md`` §4.1).  ONE traced
program serves both, because ``s`` is a traced scalar argument.

The reason this is a pair applier and not two calls is an exact algebraic
identity that is specific to LORRAX's ISDF chain.  ``encode_A`` and ``encode_B``
differ ONLY in which orbital leg carries μ and which carries ν -- Henneke Eq.
4-3's ``j_c <-> j_v`` swap sits on the encode side alone -- so they produce
tensors of IDENTICAL shape ``(μ_loc, ν_loc, ns, ns, nk)`` and IDENTICAL sharding
``P('x','y',None,None,None)``, and both then pass through the SAME convolution
and the SAME decode.  Convolution and decode are linear, so

    W_A x + s·W_B x̄  =  decode( conv( encode_A(x) + s·encode_B(x̄) ) )

-- ONE FFT pair and ONE decode for both blocks instead of two.  Against
``KERNEL_DEEPDIVE`` §3.3's byte table that turns 2 x 5.257 GB into 5.733 GB, a
predicted **1.83x**, and it is the whole reason a non-TDA step costs 2.18
TDA-matvec units rather than 4.  It is an exact identity, not an approximation
-- but it is a contraction REASSOCIATION of a sum, so it is gated at 1e-12
relative against the unfused ring appliers rather than bit-exactly (the tree's
standard for this class, cf. the ``contract_bands_block_reshard`` note in
``bse_ring_comm.py``).

The exchange term fuses the same way and for the same reason (one ``V_q0``
solve, one ``M`` decode, two encodes), which is worth 0.15% of the traffic and
is done because it falls out, not because it pays.  The diagonal ``D`` is
applied ONCE, not twice: ``B`` has no ``D`` term.
"""
from __future__ import annotations

import math
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
from jax import lax
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

from jax import shard_map as _shard_map_fn
from lxkit import mark_varying

from common.contract_bands import reduce_scatter_to_band_block
from common.fft_helpers import (klead_outer_decode_refusal, klead_outer_refusal,
                                make_local_kconv_klead, make_local_kconv_klead_outer,
                                make_local_kconv_klead_outer_decode)
from common.gpu_utils import device_budget_bytes, record_stage_price, warn_over_budget
from runtime import tiles
from .bse_preconditioner import exchange_spin_weight
from .bse_ring_comm import make_bse_shardings


def _gather_trial_block(X):
    """(b, c_loc, v_loc, nk) → the WHOLE (b, c, v, nk) trial block, once.

    The trial vector is the smallest tensor in the chain (T carries two ζ
    axes, R one) and carries no ζ axis at all, so gathering IT — once per
    block, never per trial — keeps both ζ legs stationary in every encode.
    Per rank this holds ``b·c·v·nk·16`` bytes (Si 836c at P4: 49 KB per
    trial), the one replicated operand of the W term.
    """
    return lax.all_gather(lax.all_gather(X, "y", axis=2, tiled=True),
                          "x", axis=1, tiled=True)


def _scatter_trial_block(WX, mesh_xy):
    """Σ over the mesh of the (b, c, v, nk) decode partial → (b, c_x, v_y, nk).

    The ONE reduction of the W term per block (the slab-contraction
    primitive, ``common.contract_bands``): each rank's partial holds its
    (μ_loc, ν_loc) share of both decode sums.
    """
    return reduce_scatter_to_band_block(
        WX, px=int(mesh_xy.shape["x"]), py=int(mesh_xy.shape["y"]),
        row_axis=1, col_axis=2)


# ===========================================================================
# The W term's collective schedule
# ===========================================================================
#
# The W term runs NO collective per trial (survey_C §C1, 2026-09-24).  The
# trial block X (b, c, v, nk) -- the smallest tensor in the chain, with no ζ
# axis -- is all-gathered over both mesh axes ONCE per block, both encodes
# build their partials with the full c and v locally, the decode contracts
# through this rank's μ_loc and ν_loc into a (b, c, v, nk) partial, and ONE
# reduce-scatter after the scan completes both sums
# (``common.contract_bands.reduce_scatter_to_band_block``).  This retires the
# per-trial 'x' pair (an all-gather of R and a psum_scatter of A, 426 KB per
# trial at P=64) WITHOUT the (n_trials, c, nk, ns, nu_loc) staging buffer that
# made hoisting that pair a vetoed memory-for-comm trade: the block-sized
# operands here are X-class, b·c·v·nk.  The price is the two small
# ζ-free GEMMs running over c_full instead of c_loc, (p_x-1)·v/(ns·μ_loc) of
# the encode's flops.  (Was: the 'y' pair hoisted per block since
# 2026-08-08, the 'x' pair per trial.)
#
# The permanent hoist does not change the number of live
# (nk, mu_loc, nu_loc, ns^2)-class intermediates, which stays at the one
# ``T_b`` family documented below.


# ===========================================================================
# The three stages of the W term, factored so the A block and the coupling
# block share them.  These are the SAME einsums, in the same order, that
# ``_w_stack``'s body used inline before the port -- pure code motion, so the
# shipped TDA path is bit-identical (gated: test_bse_sp_lanczos.py::
# test_stack_matvec_tda_bit_identical_after_port).
# ===========================================================================

def _encode_T_A(X_b, psi_c_X, psi_v_Y):
    """A-block ISDF encode.  ``X_b`` (c_full, v_full, nk) -> T (nk,ns,μ_loc,ns,ν_loc).

    ``T[k,t,μ,s,ν] = Σ_c ψ^X_c[k,c,t,μ] Σ_v conj(ψ^Y_v[k,v,s,ν]) X[c,v,k]``.
    T is K-LEADING: the batched ZGEMM over k writes it in this order, so no
    T-sized transpose follows (the k-minor T of 2026-09-24 cost two per trial,
    one after this GEMM and one before the decode's).
    μ rides 'x' (from ``psi_c_X``), ν rides 'y' (from ``psi_v_Y``).  The trial
    block arrives WHOLE (gathered once per block by the caller), so both ζ
    legs are produced in stationary accumulators and nothing crosses the
    mesh here (survey_C §C1, reports/gwjax_algorithmic_upgrades_2026-09-24).
    """
    R = jnp.einsum("kvsN,cvk->kcsN", jnp.conj(psi_v_Y), X_b)   # (nk,c_full,ns,ν_loc)
    return jnp.einsum("kctM,kcsN->ktMsN", psi_c_X, R)


def _encode_T_B(Xb_b, psi_c_Y, psi_v_X):
    """Coupling-block ISDF encode -- the c<->v leg swap (Henneke Eq. 4-3).

    ``T[k,t,μ,s,ν] = Σ_v ψ^X_v[k,v,t,μ] Σ_c conj(ψ^Y_c[k,c,s,ν]) X[c,v,k]``,
    ``Xb_b`` arrives WHOLE, (c_full, v_full, nk) — gathered once per block by
    the caller — so BOTH ζ legs stay stationary and nothing crosses the mesh
    here.  The legs
    swap but the SHARDING does not: μ still rides 'x' (now from
    ``psi_v_X``) and ν still rides 'y' (now from ``psi_c_Y``), so this T is
    add-compatible with ``_encode_T_A``'s with no collective and no
    resharding.  That is the fusion's precondition
    and it holds because LORRAX uses ONE ζ set for both legs (unlike Henneke's
    separate N_μ^vv / N_μ^cc / N_μ^vc).
    """
    # NEVER ring or gather a PARTIAL CONTRACTION along an axis its own ζ shard
    # lives on.  ``R`` carries ν on 'y' (it comes from ``psi_c_Y``) as well as
    # v on 'y', so ``all_gather(R, "y", axis=v)`` concatenates tiles whose ν
    # shards differ: every 'y' rank then files its neighbours' ζ tiles against
    # its own ν shard.  That was the 2026-08-08 K^d_B defect -- silent at P=1
    # (a one-rank gather is the identity) and worth two thirds of the coupling
    # correction at 2x2.  The communication goes on the TRIAL VECTOR instead,
    # which carries no ζ axis at all: gather ``Xb`` on 'y' as well as 'x', so
    # μ and ν are both produced into stationary accumulators and never travel.
    #
    # This mirrors ``bse_ring_comm._encode_T_B_gather`` / ``_ring_sum_B_encode``
    # from fix/kdb-zeta-sharding-2026-08-08 @ 443a23fe (FIX_kdb_sharding.md);
    # the port carried that file's defect here, so it takes that file's fix.
    # X is the smallest tensor in the chain -- T carries TWO ζ axes, R carries
    # one -- so this takes the T- and R-sized tensors off the wire entirely.
    R = jnp.einsum("kcsN,cvk->kvsN", jnp.conj(psi_c_Y), Xb_b)  # (nk,v_full,ns,ν_loc)
    return jnp.einsum("kvtM,kvsN->ktMsN", psi_v_X, R)


def _w_r_kminor(W_R):
    """The rank's ``W_R`` tile (μ_loc, ν_loc, kx, ky, kz) -> (μ_loc, ν_loc, nk): a reshape.

    The outer load reads the stored kernel k-minor, so its route makes no transpose
    (``_w_r_klead`` is the XLA route's, 0.88 ms and a 537 MB copy per call on CrI3 8x8).
    """
    return W_R.reshape(W_R.shape[0], W_R.shape[1], -1)


def _w_r_klead(W_R):
    """The rank's ``W_R`` tile (μ_loc, ν_loc, kx, ky, kz) -> k-leading (nk, μ_loc, ν_loc).

    Once per matvec call, outside the trial scan: callers keep building W_R
    k-minor (``make_kfft_kminor``), the conv's kernel is read k-leading.
    """
    mu_loc, nu_loc = W_R.shape[0], W_R.shape[1]
    return jnp.moveaxis(W_R.reshape(mu_loc, nu_loc, -1), -1, 0)


def _xla_conv(kconv, T_b, W_Rk):
    """``U_b = fftn_ortho(ifftn_ortho(T_b) · W_R)``: the k-convolution of an XLA-built T.

    ``(1/Nk) fft_k(W_R · ifft_k-unnormalised(T_b))`` as ONE in-place call of the
    k-convolution router's local k-LEADING factory, ``kconv``
    (``make_local_kconv_klead``: nvidia-mathdx mode 2 on CUDA, the pass Σ's τ
    kernel runs; the plan route on cpu).  Both norm factors are one folded
    constant inside it; ``W_Rk`` is ``W_R`` already in R space
    (``bse_feast.ensure_W_R``), read k-leading (``_w_r_klead``).

    THIS IS AN FFT AND IT STAYS AN FFT.  A dense (nk x nk) DFT contraction gives
    the same numbers and measured 2.3x faster on this deck, and it was REMOVED
    on 2026-07-31 by owner directive: the dense form is O(nk^2) where the FFT is
    O(nk log nk), so it is a win only because nk = 16 here and it inverts at the
    thousand-k-point sizes LORRAX is being built for.  Do not reintroduce it,
    under any name, on any measurement.  If the k-transform is a bottleneck the
    answer is a better FFT library, never a denser algorithm.
    """
    return kconv(T_b, W_Rk)                                       # in place on T_b


def _decode_tm(U_b, psi_c_X):
    """The decode's (t, μ) contraction, ``A[k,c,s,ν] = Σ_{t,μ} conj(ψ^X_c[k,c,t,μ]) U_b[k,t,μ,s,ν]``.

    ``U_b`` (nk, ns, μ_b, ns, ν_b), k-leading, and ``psi_c_X`` (nk, c_full, ns, μ_b) give
    (nk, c_full, ns, ν_b): this rank's partial over its μ block, a batched ZGEMM over k.
    (t, μ) runs first on every route, including the fused load's store: contracting the
    smaller band set first saves flops only once the decode is compute-bound (it reads U
    at ~69% of HBM today, 1.96 against 1.98 ms either order on CrI3 8x8), and the
    (s, ν)-first order moved Haydock ε₂ by 1.0e-6 against 4e-7 for this one (BSEMAX,
    claims ledger).
    """
    return jnp.einsum("kctM,ktMsN->kcsN", jnp.conj(psi_c_X), U_b)


def _decode_v(A, psi_v_Y, sqrt_nk):
    """The decode's (s, ν) contraction, ``(WX)[c,v,k] = (1/√Nk) Σ_{s,ν} ψ^Y_v[k,v,s,ν] A[k,c,s,ν]``.

    ``A`` (nk, c_full, ns, ν_b) from :func:`_decode_tm` or the fused load's store and
    ``psi_v_Y`` (nk, v_full, ns, ν_b) give (c_full, v_full, nk): this rank's partial over
    its (μ, ν) tile.  The decode contracts through this rank's μ_loc and ν_loc only; the
    caller completes both sums with one reduce-scatter per trial block after the scan, so
    no collective runs per trial.
    """
    WXcv = jnp.einsum("kvsN,kcsN->cvk", psi_v_Y, A)             # (c_full, v_full, nk)
    return WXcv / sqrt_nk


# ===========================================================================
# The TDA W term on the outer-product load (BSEMAX, 2026-09-26)
# ===========================================================================
#
# T[k,t,μ,s,ν] = Σ_c ψ^X_c Σ_v conj(ψ^Y_v) X is a rank-K sum per k with K = min(n_c, n_v), and
# it is the largest object of the matvec (2.15 GB per rank per trial on CrI3 8x8 at P4) while
# its encode does ~7 flop per byte of T written.  The router's outer factory
# (``make_local_kconv_klead_outer``) forms T in shared memory on the convolution's load from
# its two legs, ``T = Σ_K L[k,t,μ,K] R[k,K,s,ν]``, so T is never written or read: the encode
# ZGEMM and one of the convolution's two T-sized HBM passes are gone.  The legs contract over
# K = min(n_c, n_v) (8.6 against 15.0 GFLOP per trial on CrI3; the load's fp64 tensor-core
# K-sum reproduces the batched ZGEMM of the same order bit for bit).  U is unchanged
# (k-leading), and ``_decode`` reads it exactly as the XLA route does.


def _outer_legs(X_b, psi_c_X, psi_v_Y):
    """``(L, R, conj_r)`` with ``T = Σ_K L[k,t,μ,K] R'[k,K,s,ν]``, K = min(n_c, n_v),
    ``R' = conj(R)`` when ``conj_r`` else ``R``.

    n_v <= n_c: ``L = Σ_c ψ^X_c X`` (nk, ns, μ_loc, n_v), ``R' = conj(ψ^Y_v)`` read from
    ``ψ^Y_v`` itself (``conj_r``: no conjugated copy).
    n_c <  n_v: ``L = ψ^X_c`` moved K-minor (nk, ns, μ_loc, n_c), ``R' = Σ_v conj(ψ^Y_v) X``.
    """
    if psi_v_Y.shape[1] <= psi_c_X.shape[1]:
        return jnp.einsum("kctM,cvk->ktMv", psi_c_X, X_b), psi_v_Y, True
    return (jnp.moveaxis(psi_c_X, 1, -1),
            jnp.einsum("kvsN,cvk->kcsN", jnp.conj(psi_v_Y), X_b), False)


def _outer_legs_B(Xb_b, psi_c_Y, psi_v_X):
    """The coupling block's ``(L, R)``: ``T_B = Σ_K L R`` with K = min(n_c, n_v).

    ``T_B[k,t,μ,s,ν] = Σ_v ψ^X_v[k,v,t,μ] Σ_c conj(ψ^Y_c[k,c,s,ν]) Xb[c,v,k]``
    (``_encode_T_B``'s leg swap).  n_v <= n_c: ``L = ψ^X_v`` K-minor, ``R = Σ_c
    conj(ψ^Y_c) Xb``; else ``L = Σ_v ψ^X_v Xb``, ``R = conj(ψ^Y_c)``.  Same μ-on-'x',
    ν-on-'y' layout as ``_outer_legs``, so the two concatenate along K.
    """
    if psi_v_X.shape[1] <= psi_c_Y.shape[1]:
        return (jnp.moveaxis(psi_v_X, 1, -1),
                jnp.einsum("kcsN,cvk->kvsN", jnp.conj(psi_c_Y), Xb_b))
    return (jnp.einsum("kvtM,cvk->ktMc", psi_v_X, Xb_b), jnp.conj(psi_c_Y))


def _make_outer_router(mesh_xy, kgrid):
    """``route(rank) -> conv | None``: the outer kconv call when it serves this mesh and grid, else None.

    Decided at trace time and announced once per (route, K); K is the leg rank the call
    will pass (min(n_c, n_v), doubled for the fused coupling pair).
    """
    from ffi.gate import announce_once
    built = {}

    def route(rank):
        why = klead_outer_refusal(mesh_xy, kgrid)
        announce_once(("bse", "w_term_route", rank, why is None),
                      "[bse] W term: " + ("outer-product load (T never stored)" if why is None
                                          else "XLA encode + k-conv: " + why)
                      + f", K = {rank}")
        if why is not None:
            return None
        if "conv" not in built:
            built["conv"] = make_local_kconv_klead_outer(mesh_xy, kgrid, norm="ortho")
        return built["conv"]
    return route


def _w_route(route, rank, mesh_xy, kgrid, n_c):
    """``(name, conv, fused)``: the W term's route for this mesh, grid, leg rank K and band count.

    ``'fused'``: the outer-product load forms T and its store contracts U with conj(ψ^X_c)
    (``fused`` is ``make_local_kconv_klead_outer_decode``'s ``(prep, apply)``): neither T nor U
    reaches HBM.  ``'outer'``: the load forms T (``conv``) and writes U; XLA decodes it.
    ``'xla'``: XLA builds T and the k-leading convolution writes U.  Decided at trace time and
    announced once (``[bse] W term:``, ``[bse] W term decode:``).
    """
    from ffi.gate import announce_once
    conv = route(rank)
    if conv is None:
        return "xla", None, None
    why = klead_outer_decode_refusal(mesh_xy, kgrid, n_c)
    announce_once(("bse", "w_term_decode", why is None),
                  "[bse] W term decode: " + ("fused into the outer load's store (U never stored)"
                                             if why is None else "XLA on the stored U: " + why))
    if why is not None:
        return "outer", conv, None
    return "fused", conv, make_local_kconv_klead_outer_decode(mesh_xy, kgrid, norm="ortho")


def _w_layout(name):
    """The route's view of a ``W_R`` block (μ_b, ν_b, kx, ky, kz): k-minor for the outer
    load (a reshape), k-leading for the XLA route's convolution (a copy)."""
    return _w_r_klead if name == "xla" else _w_r_kminor


def _fused_decode(fused, L, R, W, pc, pc_t, conj_r):
    """The fused load's A (nk, c_full, ns, ν_b): ``pc_t`` is ψ^X_c tiled once per call
    (the whole tile), else this block's ``pc`` is tiled here."""
    prep, apply = fused
    return _rank_partial(apply(L, R, W, prep(pc) if pc_t is None else pc_t, conj_r=conj_r))


def _tda_block(name, conv, fused, kconv, pc_t):
    """``block(X_b, mu, nu, W) -> A``: one (μ, ν) block of the TDA W term, decoded through (t, μ).

    ``X_b`` (c_full, v_full, nk) is the whole trial; ``mu = (ψ^X_c,)`` (nk, c_full, ns, μ_b) and
    ``nu = (ψ^Y_v,)`` (nk, v_full, ns, ν_b) are the block's slices of the legs, ``W`` the block of
    ``W_R`` in the route's layout (:func:`_w_layout`).  Returns A (nk, c_full, ns, ν_b).  Route
    ``xla``: T by :func:`_encode_T_A`, :func:`_xla_conv`; ``outer``: T on the load from
    :func:`_outer_legs`, U stored; ``fused``: :func:`_fused_decode`.
    """
    if name == "xla":
        return lambda X_b, mu, nu, W: _decode_tm(_xla_conv(kconv, _encode_T_A(X_b, mu[0], nu[0]), W), mu[0])

    def outer_load(X_b, mu, nu, W):
        L, R, cj = _outer_legs(X_b, mu[0], nu[0])
        if name == "outer":
            return _decode_tm(conv(L, R, W, conj_r=cj), mu[0])
        return _fused_decode(fused, L, R, W, mu[0], pc_t, cj)
    return outer_load


def _pair_blocks(name, conv, fused, kconv, pc_t, sc):
    """``(block_AB, block_A, block_B)``: the coupling pair's W term on one (μ, ν) block.

    ``xs = (X_b, Xb_b)``: the whole trial and its conjugate; ``mu = (ψ^X_c, ψ^X_v)`` and
    ``nu = (ψ^Y_v, ψ^Y_c)``: the block's slices (the B encode's leg swap reads ψ^X_v on μ and
    ψ^Y_c on ν, :func:`_encode_T_B` / :func:`_outer_legs_B`).  ``block_AB`` is the fused
    applier's ONE chain of ``T_A + s·T_B`` (legs concatenated along K on the outer load);
    ``block_A`` and ``block_B`` are the unfused twin's two chains.  Each returns A
    (nk, c_full, ns, ν_b) for :func:`_blocked_w_term`.
    """
    if name == "xla":
        def chain(T, mu, W):
            return _decode_tm(_xla_conv(kconv, T, W), mu[0])

        def T_A(xs, mu, nu):
            return _encode_T_A(xs[0], mu[0], nu[0])

        def T_B(xs, mu, nu):
            return _encode_T_B(xs[1], nu[1], mu[1])
        return (lambda xs, mu, nu, W: chain(T_A(xs, mu, nu) + sc * T_B(xs, mu, nu), mu, W),
                lambda xs, mu, nu, W: chain(T_A(xs, mu, nu), mu, W),
                lambda xs, mu, nu, W: chain(T_B(xs, mu, nu), mu, W))

    def decode(L, R, cj, mu, W):
        if name == "outer":
            return _decode_tm(conv(L, R, W, conj_r=cj), mu[0])
        return _fused_decode(fused, L, R, W, mu[0], pc_t, cj)

    def block_AB(xs, mu, nu, W):
        L_A, R_A, cA = _outer_legs(xs[0], mu[0], nu[0])
        L_B, R_B = _outer_legs_B(xs[1], nu[1], mu[1])
        R_A = jnp.conj(R_A) if cA else R_A      # one convention for the concatenation
        return decode(jnp.concatenate([L_A, sc * L_B], axis=-1),
                      jnp.concatenate([R_A, R_B], axis=1), False, mu, W)

    def block_A(xs, mu, nu, W):
        L_A, R_A, cA = _outer_legs(xs[0], mu[0], nu[0])
        return decode(L_A, R_A, cA, mu, W)

    def block_B(xs, mu, nu, W):
        return decode(*_outer_legs_B(xs[1], nu[1], mu[1]), False, mu, W)
    return block_AB, block_A, block_B


# ===========================================================================
# The W term over (μ, ν) blocks
# ===========================================================================
#
# T, U and W_R are (μ, ν)-local: T is an outer product per k in its two ζ legs, the
# k-convolution acts on each (t, μ, s, ν) column with its own W_R[μ, ν, ·], and the decode
# sums over μ and ν.  So the W term is an exact sum over (μ, ν) blocks of the rank's tile
# (``_blocked_w_term``), and the block -- not the tile -- bounds what is stored per trial.
# That is what lifts the two size limits of the unblocked program: the stored U (T in place)
# of the routes that keep it is nk·ns²·μ_loc·ν_loc complex128 per trial (130 GB per rank at
# CrI3 30x30, N_μ 3000, P4), and every operand of the outer-load handlers must stay below 2^31
# elements (their 32-bit offsets; the unfused CrI3 8x8, 42v x 30c, N_μ 6000, P4 U is 2.30e9).

#: Every operand of an outer-load handler call holds fewer elements than this: the loads
#: index from per-tile 64-bit bases with 32-bit offsets (``kconv_outer_cuda_ffi.cc``,
#: "an operand holds >= 2^31 elements").  XLA and mathdx mode 2 index with 64 bits.
_HANDLER_ELEMENTS = 1 << 31


def _block_width(b, n):
    """``b`` floored to a multiple of 8 (at least 8), or the whole axis ``n`` when ``b >= n``."""
    return n if b >= n else min(n, max(8, b // 8 * 8))


def _balanced(b, n):
    """``(width, count)``: ``count = ceil(n / b)`` blocks of the narrowest multiple of 8 that
    still covers ``n``, so the clamped last block overlaps its neighbour least."""
    count = -(-n // b)
    if count == 1:
        return n, 1
    even = -(-n // count)                               # ceil(n / count)
    return min(b, -(-even // 8) * 8), count


def plan_w_blocks(route, *, nk, ns, mu_loc, nu_loc, n_c, rank):
    """The (μ, ν) blocks of one rank's W term, ``((bm, nm), (bn, nn))``, from shapes alone.

    Parameters
    ----------
    route : {'fused', 'outer', 'xla'}
        The W term's route (:func:`_w_route`).
    nk : int
        Full-zone k points of the convolution's grid.
    ns : int
        Spinor components of ψ (1 or 2).
    mu_loc, nu_loc : int
        This rank's tile of ``W_R``, ``N_μ,pad / p_x`` by ``N_μ,pad / p_y``.
    n_c : int
        Bands of the decode's left leg ψ^X_c (the TD window on the TD-aGW model).
    rank : int
        The outer load's K: ``min(n_c, n_v)``, twice that for the fused coupling pair.

    Returns
    -------
    ((bm, nm), (bn, nn)) : tuple
        Block widths and counts on μ and ν; ``nm·nn = 1`` is the whole tile.

    Every rank derives the same plan without a collective, and no result depends on the
    device budget (TASTE 96, ``docs/architecture/decisions.md#fixed-tile``):

    * The routes that store U (``outer``; ``xla``, whose T is U in place) take the largest
      square-ish block whose U block and W_R block, ``16·nk·(ns² + 1)·bm·bn`` B, fit one fixed
      tile (``runtime.tiles.TILE_BYTES``, 1 GiB); a width is a multiple of 8 (the kernels'
      8 x 8 (μ, ν) tile) or the whole axis.  The ``fused`` route stores neither T nor U (its
      handler bounds its own scratch at U/16, ns²/16 of the resident W_R tile), so its tile
      stays whole.
    * Every operand of a block's handler call -- the U block, the legs and their fragment
      tiles, the W_R block, the decoded A -- holds fewer than 2^31 elements: the wider width
      halves until it does.  An 8 x 8 block that still breaks the rule refuses
      (``GATE bse_w_block_operand``).
    """
    if route not in ("fused", "outer", "xla"):
        raise ValueError(f"plan_w_blocks: route must be 'fused', 'outer' or 'xla', got {route!r}")
    nk, ns, mu_loc, nu_loc, n_c, rank = (int(v) for v in (nk, ns, mu_loc, nu_loc, n_c, rank))
    h = -(-rank // 4)                                    # K in the load's m8n8k4 chunks

    def largest_operand(bm, bn):                         # elements of the call's largest operand
        if route == "xla":
            return 0
        if route == "outer":                             # U, L, R
            return max(nk * ns * ns * bm * bn, nk * ns * bm * 4 * h, nk * 4 * h * ns * bn)
        xb, yb, mb = -(-bm // 8), -(-bn // 8), -(-n_c // 8)   # V, Pc, L, R fragments, A
        return max(bm * bn * nk, ns * xb * nk * mb * 64, ns * xb * nk * h * 32,
                   ns * yb * nk * h * 32, nk * n_c * ns * bn)

    if route == "fused":
        bm, bn = mu_loc, nu_loc
    else:
        cells = tiles.tile_units(16 * nk * (ns * ns + 1), mu_loc * nu_loc,
                                 floor=min(64, mu_loc * nu_loc))
        bn = _block_width(math.isqrt(cells), nu_loc)
        bm = _block_width(cells // bn, mu_loc)
        bn = _block_width(cells // bm, nu_loc)           # a narrow μ tile hands its room to ν
    while largest_operand(bm, bn) >= _HANDLER_ELEMENTS:
        if bm <= 8 and bn <= 8:
            raise ValueError(
                f"GATE bse_w_block_operand: got an {bm} x {bn} (mu x nu) W-term block whose "
                f"{route} handler call holds {largest_operand(bm, bn)} elements in one operand "
                f"(nk = {nk}, ns = {ns}, n_c = {n_c}, K = {rank}); want < 2^31 = {_HANDLER_ELEMENTS} "
                f"(the outer load's 32-bit offsets); why: no block of this k grid fits the handler; "
                f"fix: a coarser k grid, fewer bands in the window, or the XLA route "
                f"(docs/architecture/bse.md#the-matvec)")
        if bm >= bn:
            bm = _block_width((bm + 1) // 2, mu_loc)
        else:
            bn = _block_width((bn + 1) // 2, nu_loc)
    return _balanced(bm, mu_loc), _balanced(bn, nu_loc)


def _w_plan(name, rank, W_R, mu, nu):
    """The trace's block plan (:func:`plan_w_blocks`), announced once and priced.

    The price is the W term's per-rank live set: the resident ``W_R`` tile and ψ legs, and per
    trial the route's block transients (the stored U, the laid-out ``W_R`` block, A and its
    accumulator, the outer load's legs, the fused handler's scratch at most U/16).  It is
    recorded (``record_stage_price``) and, over ``device_budget_bytes()``, warned
    (``warn_over_budget``); it never sizes the blocks.
    """
    from ffi.gate import announce_once
    mu_loc, nu_loc = int(W_R.shape[0]), int(W_R.shape[1])
    nk = int(np.prod(W_R.shape[2:]))
    n_c, ns = int(mu[0].shape[1]), int(mu[0].shape[2])
    plan = plan_w_blocks(name, nk=nk, ns=ns, mu_loc=mu_loc, nu_loc=nu_loc, n_c=n_c, rank=rank)
    (bm, nm), (bn, nn) = plan
    whole = nm * nn == 1
    cells = 16 * nk * bm * bn                                      # one complex128 (μ, ν, k) block
    resident = 16 * nk * mu_loc * nu_loc + 16 * sum(int(np.prod(a.shape)) for a in mu + nu)
    stored = 0 if name == "fused" else ns * ns * cells             # U (T in place on 'xla')
    layout = 0 if (whole and name != "xla") else cells             # the laid-out W_R block
    decoded = 16 * nk * n_c * ns * bn * (1 if whole else 2)        # A and its accumulator
    legs = 0 if name == "xla" else 16 * nk * ns * 4 * -(-rank // 4) * (bm + bn)
    scratch = ns * ns * cells // 16 if name == "fused" else 0
    price = resident + stored + layout + decoded + legs + scratch
    gb = lambda v: f"{v / 1e9:.3g} GB"
    announce_once(("bse", "w_term_blocks", name, plan),
                  "[bse] W term blocks: " + (f"whole tile {mu_loc} x {nu_loc} (mu_loc x nu_loc)"
                                             if whole else
                                             f"{nm} x {nn} blocks of {bm} x {bn} in the {mu_loc} x "
                                             f"{nu_loc} (mu_loc x nu_loc) tile")
                  + (f", U block {gb(stored)} (tile {gb(tiles.TILE_BYTES)})" if stored else "")
                  + f"; W term {gb(price)}/rank")
    record_stage_price("bse.w_term", price)
    budget = device_budget_bytes()
    if price > budget:
        warn_over_budget("bse.w_term", price, budget)
    return plan


def _take_block(arrays, i, b, n):
    """Block ``i`` (traced) of width ``b`` along the LAST axis, of length ``n``, of each array; and its start.

    The block starts at ``min(i·b, n − b)``: the last block is clamped to end at ``n``, and the
    rows it shares with block ``i − 1`` (those below ``i·b``) are zeroed, so every row of the
    axis is live in exactly one block.
    """
    start = jnp.minimum(i * b, n - b)
    keep = jnp.arange(b) >= i * b - start
    return tuple(jnp.where(keep, lax.dynamic_slice_in_dim(a, start, b, axis=a.ndim - 1), 0)
                 for a in arrays), start


def _w_block(W_R, m0, n0, bm, bn):
    """The (μ, ν) block of the rank's ``W_R`` tile at ``(m0, n0)``, ``(bm, bn, kx, ky, kz)``.

    Sliced from the tile's 2-D ``(μ_loc, ν_loc·nk)`` view, so that no layout of the sliced array
    turns the block's k-leading relayout (:func:`_w_r_klead`) into a bitcast.  Sliced 5-D, XLA's
    GPU layout assignment meets the k-leading kernel operand by giving the loop-invariant tile a
    k-major layout ahead of the scans -- one transposed copy of the WHOLE tile, 32.4 GB per rank at
    CrI3 30x30, N_μ 3000, P4, the module the AOT gate refused at 98.5 GB (buffer assignment,
    2026-10-16).  From the 2-D view the transpose stays a block-sized copy inside the step.
    """
    mu_loc, nu_loc = int(W_R.shape[0]), int(W_R.shape[1])
    kshape = tuple(int(n) for n in W_R.shape[2:])
    nk = math.prod(kshape)
    flat = W_R.reshape(mu_loc, nu_loc * nk)
    return lax.dynamic_slice(flat, (m0, n0 * nk), (bm, bn * nk)).reshape((bm, bn) + kshape)


def _blocked_w_term(plan, W_R, w_layout):
    """``wterm(mu, nu, block, sqrt_nk) -> (WX)_b``: this rank's (c_full, v_full, nk) W-term
    partial of one trial, summed over the (μ, ν) blocks of ``plan``.

    THE W TERM SPLITS EXACTLY OVER (μ, ν) BLOCKS.  For blocks I_i of μ and J_j of ν,

        (WX)[c,v,k] = Σ_j (1/√Nk) Σ_{s, ν∈J_j} ψ^Y_v[k,v,s,ν] A_j[k,c,s,ν],
        A_j[k,c,s,ν] = Σ_i Σ_{t, μ∈I_i} conj(ψ^X_c[k,c,t,μ]) U_ij[k,t,μ,s,ν],

    U_ij being the k-convolution of T's (I_i, J_j) block with W_R's: only the order of the μ
    and ν sums changes (round-off).  ``block(mu_i, nu_j, W_ij)`` returns A's (i, j) term;
    ``mu`` and ``nu`` are tuples of arrays whose LAST axis is μ and ν (the ψ legs, the first of
    each the decode's own); ``W_R`` is the rank's (μ_loc, ν_loc, kx, ky, kz) tile and
    ``w_layout`` the route's view of a block of it (:func:`_w_layout`).

    ``plan = ((bm, nm), (bn, nn))`` from :func:`plan_w_blocks`.  The whole tile (one block) is
    the unblocked program itself: ``W_R`` laid out once per call, outside the trial scan, and
    ``block`` on the whole legs.  Otherwise a ``lax.scan`` over ν blocks carries the
    (c, v, nk) partial and a nested ``lax.scan`` over μ blocks carries A_j, both
    ``unroll=1``: one compiled body whatever the block count (13 x 13 per trial at CrI3
    30x30, N_μ 3000, P4, growing as N_μ²/P), each block's ``W_R`` slice
    (:func:`_w_block`) laid out inside its step.  The last block of an axis is clamped and its
    shared rows zeroed in every leg (:func:`_take_block`), so no operand is padded.
    """
    (bm, nm), (bn, nn) = plan
    if nm * nn == 1:
        W = w_layout(W_R)                          # once per call, outside the trial scan
        return lambda mu, nu, block, sqrt_nk: _decode_v(block(mu, nu, W), nu[0], sqrt_nk)
    mu_loc, nu_loc = int(W_R.shape[0]), int(W_R.shape[1])

    def wterm(mu, nu, block, sqrt_nk):
        nk, n_c, ns = (int(n) for n in mu[0].shape[:3])
        n_v = int(nu[0].shape[1])

        def nu_step(WX, j):
            nu_j, n0 = _take_block(nu, j, bn, nu_loc)

            def mu_step(A, i):
                mu_i, m0 = _take_block(mu, i, bm, mu_loc)
                W_ij = w_layout(_w_block(W_R, m0, n0, bm, bn))
                return A + _rank_partial(block(mu_i, nu_j, W_ij)), None

            A0 = mark_varying(jnp.zeros((nk, n_c, ns, bn), mu[0].dtype), ("x", "y"))
            A, _ = lax.scan(mu_step, A0, jnp.arange(nm), unroll=1)
            return WX + _decode_v(A, nu_j[0], sqrt_nk), None

        WX0 = mark_varying(jnp.zeros((n_c, n_v, nk), mu[0].dtype), ("x", "y"))
        WX, _ = lax.scan(nu_step, WX0, jnp.arange(nn), unroll=1)
        return WX
    return wterm


def _rank_partial(A):
    """The fused kconv call's ``A`` is this rank's (μ_loc, ν_loc) partial: varying over 'x' and 'y'.

    The custom call's result carries only the varying axes jax infers for it (the XLA route
    gets 'x' from the einsum with ψ^X_c), so the missing ones are declared here.
    """
    have = getattr(getattr(jax, "typeof", lambda a: None)(A), "vma", None) or frozenset()
    miss = tuple(a for a in ("x", "y") if a not in have)
    return mark_varying(A, miss) if miss else A


def _exchange_U(S_part, V_q0):
    """Inside the (x, y) shard_map: the exchange term's two small reductions.

    ``S_part`` (b, μ) is this rank's partial of the k-summed encode, contracted
    over its own (c, v) transition tile; ``V_q0`` its (μ_x, ν_y) tile.  One psum
    completes S, the tile product gives U's μ_x rows summed over this ν_y block,
    and a second psum completes and replicates U = V_q0 · S, (b, μ).  Both
    psums move a (b, μ) vector -- the V term's only collectives.
    """
    S = lax.psum(S_part, ("x", "y"))
    mx, ny = V_q0.shape
    S_y = lax.dynamic_slice_in_dim(S, lax.axis_index("y") * ny, ny, axis=1)
    U_x = jnp.einsum("MN,bN->bM", V_q0, S_y)                    # (b, μ_x) partial in ν
    U = lax.dynamic_update_slice_in_dim(
        jnp.zeros_like(S), U_x, lax.axis_index("x") * mx, axis=1)
    return lax.psum(U, ("x", "y"))


def _make_v_term(mesh_xy, pair: bool):
    """The exchange (V) term on the transition layout, one shard_map.

    ``pair=False``: ``f(X, M, V_q0, enc, dec) = dec · M (V_q0 (enc · M† X))``.
    ``pair=True``:  ``f(X, Xb, sc, M, V_q0, enc, dec)`` adds the coupling
    block's bare-vertex encode ``sc · M^T Xb`` before the ONE V_q0 solve.
    X, Xb and M are all (c on x, v on y), so both encodes and the decode are
    rank-local; ``enc``/``dec`` are the scalar normalisations.
    """
    xspec = P(None, "x", "y", None)

    def _tda(X, M, V_q0, enc, dec):
        S = jnp.einsum("kcvN,bcvk->bN", jnp.conj(M), X) * enc      # (b, μ) partial
        return jnp.einsum("kcvM,bM->bcvk", M, _exchange_U(S, V_q0)) * dec

    def _pair(X, Xb, sc, M, V_q0, enc, dec):
        S = (jnp.einsum("kcvN,bcvk->bN", jnp.conj(M), X)
             + sc * jnp.einsum("kcvN,bcvk->bN", M, Xb)) * enc
        return jnp.einsum("kcvM,bM->bcvk", M, _exchange_U(S, V_q0)) * dec

    if pair:
        return _shard_map_fn(_pair, mesh=mesh_xy,
                             in_specs=(xspec, xspec, P(), xspec, P("x", "y"), P(), P()),
                             out_specs=xspec, check_vma=False)
    return _shard_map_fn(_tda, mesh=mesh_xy,
                         in_specs=(xspec, xspec, P("x", "y"), P(), P()),
                         out_specs=xspec, check_vma=False)


def build_bse_stack_matvec(
    mesh_xy: Mesh,
    nkx: int,
    nky: int,
    nkz: int,
    *,
    kernel: str = "bse",
    head_tensor: bool = False,
):
    """Build the trial-stack BSE matvec.

    Parameters
    ----------
    kernel : {'bse', 'rpa'}
        ``'bse'`` returns ``D + V - W`` (screened direct term); ``'rpa'`` returns
        ``D + V`` (the W-term ``shard_map`` is not built).
    head_tensor : bool
        Add the cell-averaged nonanalytic exchange head as a rank-three term
        over the TRANSITION index, and take two extra runtime arguments
        ``(D_head, M_head)`` to carry it.  Default False traces a program with
        no head contraction in it at all — not a zero-valued one — so the
        off path is bit-identical, exactly as ``W_q0=None`` does for the
        screened-direct term in ``bse_davidson_helpers``.

        THE HEAD CANNOT LIVE IN THE μ BASIS, WHICH IS WHY IT IS A SEPARATE
        TERM.  The exchange tile's head channel is rank one in μ with a
        SCALAR Coulomb coefficient, and the object the cell average actually
        needs is ``M_ab = <v(q) q_a q_b>_cell`` — a tensor, whose contraction
        through μ would need ``∂_a ζ̃_μ`` and would break ``eval_vq``'s
        ``A = zt·√v`` factorisation.  It does not have to: the head's
        q-linear coefficient is the transition dipole, so

            K^head_{t,t'} = (1/N_k) · conj(d_a(t)) · M_ab · d_b(t')

        is rank three over transitions and belongs beside ``M``,
        where this matvec already carries rank-three objects
        (``LT_HEAD_PROBLEM.md`` §6).

        Structurally it IS the exchange term with ``(M, M, V_q0)``
        replaced by ``(D_head, D_head, M_head)``, which is why it reuses the
        same encode/decode shape and the same ``1/N_k``.  ``D_head`` is
        ``conj(d)`` with the same ``(k, c, v, a)`` layout as ``M`` and a
        Cartesian axis of length 3 where μ was; ``M_head`` is the real
        symmetric ``(3, 3)`` cell moment.  Hermiticity of the added term is
        then automatic: ``M`` real symmetric ⇒ ``K^head`` Hermitian.
    """
    if kernel not in ("rpa", "bse"):
        raise ValueError(f"kernel must be 'rpa' or 'bse', got {kernel!r}")
    include_W = kernel == "bse"

    sh = make_bse_shardings(mesh_xy)
    nk = nkx * nky * nkz
    # The W-term k-convolution: the router's outer-product factory when it serves
    # the band rank K = min(n_c, n_v) (decided at trace time, announced once),
    # else the local k-leading factory on an XLA-built T; one call per trial.
    kconv = make_local_kconv_klead(mesh_xy, (nkx, nky, nkz), norm="ortho")
    outer_route = _make_outer_router(mesh_xy, (nkx, nky, nkz))

    # ── W term: one shard_map over ('x','y'); body = scan over the trial axis ──
    def _w_stack(X, psi_c_X, psi_v_Y, W_R):
        # Local shards: X (n_trials, c_loc, v_loc, nk); psi_c_X (nk, c_full, ns,
        # μ_loc); psi_v_Y (nk, v_full, ns, ν_loc); W_R (μ_loc, ν_loc, kx,ky,kz).
        # sqrt_nk follows the input dtype (fp32/fp64) — drop-in for fp32 GMRES.
        # DTYPE SEAM: this whole W-term inherits X's dtype, so a complex64 matvec
        # would halve the 655 MB T-tensor and every one of its ~4 HBM round-trips
        # (the audit's measured ~2× bandwidth lever, JOINT_FINDINGS §4). It is
        # DELIBERATELY left at complex128 (no c64 here) per owner decision
        # (2026-07-16); the fp32-GMRES path casts upstream in bse_feast, not here.
        sqrt_nk = jnp.sqrt(jnp.asarray(nk, dtype=X.real.dtype))

        # The route (outer load fused / stored U, or XLA T), then its (μ, ν) block plan:
        # both decided at trace time from the shapes and announced once.
        rank = min(psi_c_X.shape[1], psi_v_Y.shape[1])
        name, conv, fused = _w_route(outer_route, rank, mesh_xy, (nkx, nky, nkz), psi_c_X.shape[1])
        plan = _w_plan(name, rank, W_R, (psi_c_X,), (psi_v_Y,))
        wterm = _blocked_w_term(plan, W_R, _w_layout(name))
        whole = plan[0][1] * plan[1][1] == 1
        block = _tda_block(name, conv, fused, kconv,
                           fused[0](psi_c_X) if name == "fused" and whole else None)

        def _body(carry, X_b):                       # X_b: (c_full, v_full, nk)
            # NO COLLECTIVE IN THIS BODY (survey_C §C1): X_b arrives whole, and
            # every block's decode returns this rank's (μ_loc, ν_loc) partial.
            return carry, wterm((psi_c_X,), (psi_v_Y,), partial(block, X_b), sqrt_nk)

        # Once per block of trials: gather, scan, reduce once.  The partial the
        # scan stacks is (n_trials, c, v, nk), X-sized, never a T-class buffer.
        _, WX = lax.scan(_body, None, _gather_trial_block(X), unroll=1)
        return _scatter_trial_block(WX, mesh_xy)

    w_stack = _shard_map_fn(
        _w_stack,
        mesh=mesh_xy,
        in_specs=(P(None, "x", "y", None), P(None, None, None, "x"),
                  P(None, None, None, "y"), P("x", "y", None, None, None)),
        out_specs=P(None, "x", "y", None),
    )

    v_term = _make_v_term(mesh_xy, pair=False)

    def _matvec(X, psi_c_X, psi_c_Y, psi_v_X, psi_v_Y, eps_c, eps_v, W_R, V_q0,
                M, D_head=None, M_head=None):
        # M (transition layout, sh.M): the hoisted exchange pair amplitude,
        # precomputed once per solve (audit P3). psi_c_Y / psi_v_X are unused
        # here — kept for a uniform matvec signature with the ring paths;
        # psi_c_X / psi_v_Y feed the W-term.
        sqrt_nk = jnp.sqrt(jnp.asarray(nk, dtype=X.real.dtype))
        # ── D term: (ε_c − ε_v) · X  (batched, local) ──────────────────────────
        delta_E = eps_c.T[None, :, None, :] - eps_v.T[None, None, :, :]
        D_term = lax.with_sharding_constraint(delta_E * X, sh.X)

        # ── V term: B1 dense exchange, k-summed encode + broadcast decode ──────
        # K^x = M V M†: conjugated vertex on the encode leg, bare vertex on the
        # decode.  Fixed by the transition density <0|ρ̂|Ψ> = Σ A_cvk ψ_ck ψ*_vk;
        # the reverse assignment builds conj(K^x), which cannot be covariant
        # alongside the (correct, untouched) W term.
        # The encode carries the scalar-singlet weight (2 scalar, 1 spinor).
        # Encode k-SUMMED S = M† X, U = V_q0 S, decode M U broadcast over k —
        # all against the rank's own (c, v) tile; ``_exchange_U`` holds the two
        # (b, μ) psums.
        w_x = exchange_spin_weight(psi_c_X.shape[2])
        VX = v_term(X, M, V_q0, w_x / sqrt_nk, 1.0 / sqrt_nk)

        if head_tensor:
            # ── Head term: the SAME contraction with (D_head, M_head) in
            #    place of (M, V_q0, M).  Three Cartesian components stand
            #    where μ stood, so this is three inner products per trial
            #    vector and a 3x3 — free next to everything above.  The
            #    conjugation follows the V term's exactly, and must: the
            #    encode leg carries the conjugated vertex.
            # D_head = conj(d), so conj(D_head) is the bare dipole and the
            # two legs read exactly as the encode / decode M do above.
            Sh = jnp.einsum("kcva,bcvk->ba", jnp.conj(D_head), X) / sqrt_nk * w_x
            Uh = Sh @ M_head.astype(Sh.dtype).T                   # U_a = M_ab S_b
            HX = jnp.einsum("kcva,ba->bcvk", D_head, Uh)
            VX = VX + lax.with_sharding_constraint(HX, sh.X) / sqrt_nk

        if not include_W:
            return D_term + VX

        WX = w_stack(X, psi_c_X, psi_v_Y, W_R)
        return D_term + VX - WX

    in_sh = [sh.X, sh.psi_x, sh.psi_y, sh.psi_x, sh.psi_y,
             sh.eps, sh.eps, sh.W, sh.V, sh.M]
    if head_tensor:
        # D_head / M_head are small and replicated: three Cartesian channels
        # over (k, c, v) and a 3x3.  No mesh axis to tile them on.
        in_sh += [None, None]
    return jax.jit(
        _matvec,
        in_shardings=tuple(in_sh),
        out_shardings=sh.X,
    )


# ===========================================================================
#  The non-TDA pair applier — SDY Algorithm 4's F and G, fused
# ===========================================================================

def build_bse_stack_pair_matvec(
    mesh_xy: Mesh,
    nkx: int,
    nky: int,
    nkz: int,
    *,
    kernel: str = "bse",
    fuse: bool = True,
):
    """Build the real-linear pair applier ``pair(X, s, …) = A·X + s·B·conj(X)``.

    ``s = +1`` is Shao-da Jornada-Yang's ``F(x) = Ax + Bx̄``; ``s = -1`` is their
    ``G(v) = Av − Bv̄`` (Algorithm 4, arXiv:1611.02348).  ``s`` is a TRACED
    scalar, so ONE compiled program serves both halves of an SDY step and the
    compile count does not depend on how many steps run.

    Note ``F`` and ``G`` are **real**-linear only: ``F(αx) = αAx + ᾱBx̄`` is not
    ``αF(x)`` for complex ``α``.  Every consumer of this callable must therefore
    keep its Gram-Schmidt coefficients real where they ride the ``U`` basis
    (``solvers.bse_sp_lanczos`` does; see its two-coefficient reorthogonalisation).

    Parameters
    ----------
    kernel : {'bse', 'rpa'}
        ``'bse'`` returns ``D + V − W`` on the A block and ``V − W`` on the
        coupling block; ``'rpa'`` drops the screened-direct term from both.
    fuse : bool
        ``True`` (the default, and the point of this builder) sums the two
        encodes before ONE convolution and ONE decode.  ``False`` is the
        UNFUSED TWIN: two independent conv+decode chains, summed at the end.
        It is value-identical to 1e-12 and ~1.83x more expensive, and it exists
        so the fusion identity can be gated and priced against something rather
        than asserted.  Do not ship ``fuse=False``.
    """
    if kernel not in ("rpa", "bse"):
        raise ValueError(f"kernel must be 'rpa' or 'bse', got {kernel!r}")
    include_W = kernel == "bse"

    sh = make_bse_shardings(mesh_xy)
    rep = NamedSharding(mesh_xy, P())
    nk = nkx * nky * nkz
    kconv = make_local_kconv_klead(mesh_xy, (nkx, nky, nkz), norm="ortho")
    outer_route = _make_outer_router(mesh_xy, (nkx, nky, nkz))

    # ── W term: one shard_map over ('x','y') — the SAME single region the TDA
    #    stack matvec opens.  No new shard_map is created by the coupling port:
    #    the B encode is an einsum pair plus one all_gather INSIDE this body.
    def _w_pair(X, sc, psi_c_X, psi_c_Y, psi_v_X, psi_v_Y, W_R):
        # Local shards: X (n_trials, c_loc, v_loc, nk); psi_*_X (…, μ_loc);
        # psi_*_Y (…, ν_loc); W_R (μ_loc, ν_loc, kx, ky, kz).
        sqrt_nk = jnp.sqrt(jnp.asarray(nk, dtype=X.real.dtype))

        # ONE block-level gather, hoisted OUT of the scan: both encodes take
        # the whole trial block, and Xb = conj(X) is formed locally from it
        # (conj commutes with the gather exactly, so this is the same array
        # a second gather of Xb would deliver).  No collective runs per trial.
        X_full = _gather_trial_block(X)                     # (b, c, v, nk)
        Xb_full = jnp.conj(X_full)

        # THE FUSION.  T^A and T^B have identical shape AND identical sharding
        # (the B encode swaps legs, not layouts), so their sum needs no collective
        # and no reshard; conv and decode are linear, so one chain serves both
        # blocks.  On the outer-product load (BSEMAX) the fused chain concatenates
        # the two blocks' legs along K, so ONE convolution call forms T_A + s T_B in
        # shared memory.  The twin runs two chains, summed at the end.
        rank = min(psi_c_X.shape[1], psi_v_Y.shape[1])
        k_rank = 2 * rank if fuse else rank
        name, conv, fused = _w_route(outer_route, k_rank, mesh_xy, (nkx, nky, nkz), psi_c_X.shape[1])
        mu, nu = (psi_c_X, psi_v_X), (psi_v_Y, psi_c_Y)
        plan = _w_plan(name, k_rank, W_R, mu, nu)
        wterm = _blocked_w_term(plan, W_R, _w_layout(name))
        whole = plan[0][1] * plan[1][1] == 1
        block_AB, block_A, block_B = _pair_blocks(
            name, conv, fused, kconv, fused[0](psi_c_X) if name == "fused" and whole else None, sc)

        def _body_fused(carry, xs):
            return carry, wterm(mu, nu, partial(block_AB, xs), sqrt_nk)

        def _body_unfused(carry, xs):
            # THE TWIN.  Two full chains.  Kept only to price the fusion.
            return carry, (wterm(mu, nu, partial(block_A, xs), sqrt_nk)
                           + sc * wterm(mu, nu, partial(block_B, xs), sqrt_nk))

        _, WX = lax.scan(_body_fused if fuse else _body_unfused,
                         None, (X_full, Xb_full), unroll=1)
        return _scatter_trial_block(WX, mesh_xy)

    w_pair = _shard_map_fn(
        _w_pair,
        mesh=mesh_xy,
        in_specs=(P(None, "x", "y", None), P(),
                  P(None, None, None, "x"), P(None, None, None, "y"),
                  P(None, None, None, "x"), P(None, None, None, "y"),
                  P("x", "y", None, None, None)),
        out_specs=P(None, "x", "y", None),
    )

    v_pair = _make_v_term(mesh_xy, pair=True)

    def _pair(X, s, psi_c_X, psi_c_Y, psi_v_X, psi_v_Y, eps_c, eps_v, W_R,
              V_q0, M):
        sqrt_nk = jnp.sqrt(jnp.asarray(nk, dtype=X.real.dtype))
        sc = s.astype(X.dtype)
        Xb = jnp.conj(X)

        # ── D term — the A block ONLY.  B carries no diagonal, so an SDY step
        #    applies D twice per step (once in F, once in G), not four times.
        delta_E = eps_c.T[None, :, None, :] - eps_v.T[None, None, :, :]
        D_term = lax.with_sharding_constraint(delta_E * X, sh.X)

        # ── V term: both exchange encodes, one V_q0 solve, one decode ─────────
        # A: K^x  = M V M†  — CONJUGATED vertex on the encode leg (the settled
        #    B1 result, bse_stack_matvec's shipped TDA form).
        # B: K^x_B = M V M^T — the BARE vertex on the encode leg (Henneke
        #    Eq. 2-20's conjugated pairing ⟨M_t|v|conj(M_t')⟩; the ring path
        #    spells the same thing as ``apply_V_ring_B``, which conjugates ψ^Y
        #    on the way in).  DO NOT "improve" this conjugation: the exchange
        #    conjugation is settled and re-litigating it is a known failure.
        #    Both encodes and the decode are local on the transition layout
        #    (``_make_v_term``); one V_q0 solve serves both blocks.
        VX = v_pair(X, Xb, sc, M, V_q0,
                    exchange_spin_weight(psi_c_X.shape[2]) / sqrt_nk, 1.0 / sqrt_nk)

        if not include_W:
            return D_term + VX

        WX = w_pair(X, sc, psi_c_X, psi_c_Y, psi_v_X, psi_v_Y, W_R)
        return D_term + VX - WX

    return jax.jit(
        _pair,
        in_shardings=(sh.X, rep, sh.psi_x, sh.psi_y, sh.psi_x, sh.psi_y,
                      sh.eps, sh.eps, sh.W, sh.V, sh.M),
        out_shardings=sh.X,
    )
