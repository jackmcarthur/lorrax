"""Batched trial-stack BSE matvec — one T-tensor alive regardless of n_trials.

``build_bse_stack_matvec`` returns a jitted

    matvec(X[n_trials, c, v, k], psi_c_X, psi_c_Y, psi_v_X, psi_v_Y,
           eps_c, eps_v, W_R, V_q0, M_X, M_Y)  ->  out[n_trials, c, v, k]

for the TDA BSE (or RPA) Hamiltonian ``H = D + V - W`` (``D + V`` for RPA).

The exchange pair amplitudes ``M_X`` (μ on x) and ``M_Y`` (ν on y) are pure
functions of ψ (``compute_pair_amplitude``), so they are HOISTED to matvec inputs
(precomputed ONCE per solve at load time, ``bse_io``) rather than rebuilt inside
every iteration — the matvec is a per-iteration black-box jit whose ψ args XLA
cannot hoist across calls (audit P3, ``reports/bse_refactor_map_2026-07-15/archive/
matvec_efficiency_audit``). Peak-neutral (both M's already lived inside every call);
only the between-matvec floor rises by ~2·M/p. ``psi_c_Y``/``psi_v_X`` are retained
in the signature for a uniform calling convention with the ring matvecs (they now
feed only the W-term's ``psi_c_X``/``psi_v_Y``; the V-term reads the hoisted M's).

Why a stack matvec.  The four legacy TDA matvecs (ring/gather/simple/serial)
carry the trial axis ``b`` on the direct-term tensor ``T[b, μ, ν, t, s, k]`` —
per device ``n_trials · μ_loc · ν_loc · ns² · nk`` complex128, LINEAR in
``n_trials`` (the memory hog).  Here the W-term body is a ``lax.scan`` over the
trial axis, so XLA reuses the body's scratch across iterations: exactly ONE
``T``-family is alive regardless of ``n_trials``.  A Python-unrolled or
``fori_loop``-over-trials-inside-``jit`` would pile up ``n_trials`` live ``T``
slots (the known slot-pile-up failure mode,
``feedback_path_d_scaffolding_pattern``); the scan avoids it.

THE ``lax.scan`` IS WHAT BOUNDS ``T``, NOT THE ``shard_map`` — this paragraph
used to credit the wrong one.  Measured 2026-08-08 (SHARDMAP_AUDIT.md §4.4,
§6.1): the GSPMD twin below runs the same scan with NO ``shard_map`` at all and
holds 450.01 MiB at ``n_trials=1`` against 450.10 MiB at ``n_trials=8`` — flat
in the trial axis, exactly like the manual route.  Dropping the ``shard_map``
does not bring the memory hog back.

Why the manual ``shard_map`` is kept anyway, which is the justification that
actually survives measurement:

  1. BACKEND PORTABILITY of the decode collectives.  On a CPU mesh the SPMD
     partitioner emits ``all-reduce`` where the manual body issues
     ``psum_scatter`` — 2x the wire bytes on BOTH decode legs, with no flag in
     this build that fixes it.
  2. A GUARANTEE RATHER THAN A COINCIDENCE.  On GPU today the partitioner
     reproduces the manual plan exactly (same 6 collectives, same bytes), but
     that is a property of this XLA build, not a contract it owes us.  The
     manual spelling cannot silently regress.

A site that outlives its stated reason is how habit becomes doctrine, so the
stated reason is now the one the measurement supports.

Exchange (V) is the B1 dense form (VERDICT.md): DENSE in (k,k'), encode k-SUMMED
into a k-free ζ-space density, decode broadcast at every k.  ``S,U`` are k-free
(tiny, ``n_trials × ν``) so the V term stays outside the scan, batched.

Shardings (``make_bse_shardings``) are unchanged; ``n_trials`` occupies the
leading axis of ``sh.X = P(None,'x','y',None)`` that block ``b`` used to.

The W-tile seam is the single line ``U = fft_k(W_R * ifft_k(T))``: ``W_R`` is a
shape-stable ``(μ_pad, ν_pad, nkx, nky, nkz)`` argument built ONCE outside the
matvec, so W(ω) / ladder buildouts pass a different ``W_R`` with no change to
encode/decode/scan.

Retirement plan (NOTED, not yet executed) — the ring/gather/simple TDA matvecs
existed only to bound ``T``'s peak; this scan bounds it strictly better, so they
are superseded.  Consumers repointed here: ``bse_lanczos.solve_bse_sharded``
(block-Lanczos + Davidson) and ``bse_feast`` (TDA GMRES contour solves +
``_rayleigh_ritz`` subspace application).  Still live pending a follow-up:
  * ``bse_ring_comm.build_bse_ring_matvec`` — used by
    ``bse_feast.estimate_spectral_bounds_sharded`` (spectral-bound Lanczos) and
    the equality gates.  Repoint + delete together with ``bse_simple`` and the
    ``matvec_kind`` data key once the spectral-bound Lanczos is moved over.
  * ``bse_ring_comm.build_bse_ring_matvec_full`` (non-TDA S=[[A,B],[-B†,-A†]]):
    the B-encode is now PORTED HERE (``build_bse_stack_pair_matvec``, 2026-08-08),
    which is what that retirement note asked for -- the coupling block reuses
    this module's encode/decode rather than its own.  The ring full matvec stays
    live as the ``_materialize_A_B`` oracle and as the equality gate's twin.

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
solve, one ``M_X`` decode, two encodes), which is worth 0.15% of the traffic and
is done because it falls out, not because it pays.  The diagonal ``D`` is
applied ONCE, not twice: ``B`` has no ``D`` term.
"""
from __future__ import annotations

import os

import jax
import jax.numpy as jnp
import numpy as np
from jax import lax
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

from common.shard_map import shard_map as _shard_map_fn

from common.fft_helpers import (
    local_fftn3,
    make_sharded_fftn_3d,
    make_sharded_ifftn_3d,
)
from .bse_ring_comm import make_bse_shardings


# ===========================================================================
# LORRAX_BSE_MATVEC_OPT — the ONE dial for this kernel's two measured levers
# ===========================================================================
#
# Grammar: a comma list of tokens from ``_MATVEC_OPTS``; empty/unset = none.
# An UNKNOWN token REFUSES.  That refusal is the point: this project has
# already shipped a flag (``LORRAX_FFT_FFI_FUSED``) whose consumer accepted
# ``=yes`` and silently ignored ``=Y``, so a run could be labelled "optimised"
# and be running the baseline.  A perf dial that can be misspelled into a
# no-op makes every A/B built on it void.
#
#   [densek REMOVED 2026-07-31 — owner directive, no exceptions.]  It replaced
#           the k-space FFT pair with a dense (nk x nk) DFT contraction.  It
#           was correct and measured 2.30x on the matvec, and that is exactly
#           why it is gone: dense is O(nk^2) against the FFT's O(nk log nk),
#           so it wins only because this deck has nk = 16 and it inverts at
#           the thousand-k-point sizes LORRAX targets.  Do not reintroduce a
#           DFT-as-GEMM path under any name or on any measurement.  If the
#           k-transform is hot, fix the FFT (batching, an FFI handler, fewer
#           dispatches) -- 331 ns per four-point transform is dispatch
#           overhead, not arithmetic.
#
# The 'y'-axis collective hoist is PERMANENT (was ``yhoist``, made
# unconditional 2026-08-08).  ``all_gather(X_b,'y')`` and the final
# ``psum_scatter(...,'y')`` are the two SMALL collectives -- the operand is
# (c_loc, v, nk), 2 KB per trial at P=64 against 426 KB for the 'x' pair -- so
# batching them over the trial axis costs 16 KB per rank in total and removes
# 2 of the 4 collectives per trial.  The 'x' pair is deliberately NOT hoisted:
# batching those needs an (n_trials, c, nk, ns, nu_loc) staging buffer, 3.4 MB
# per rank, an n_trials-fold replication of a T-adjacent intermediate -- the
# memory-for-comm trade the owner has vetoed.  The accounting is per-rank
# bytes, and it is the whole argument for why one half of this is allowed and
# the other is not.
#
#   gspmd   AUDIT ROUTE, default OFF.  Build the W term with NO ``shard_map``:
#           the same einsum chain and the same ``lax.scan`` over trials, but
#           expressed on GLOBAL arrays with ``with_sharding_constraint`` hints
#           at each of the four points where the manual body issues a
#           collective, letting XLA's SPMD partitioner choose the collective.
#           Built for the 2026-08-08 shard_map audit to answer whether the
#           manual all_gather/psum_scatter schedule is EARNED or HABIT.
#           It is an A/B instrument, not a proposed default: it changes which
#           collectives the program issues, so every claim about it must be
#           backed by an HLO diff and a timing pair.  See SHARDMAP_AUDIT.md.
#           NOTE the structural consequence, which is the reason it exists:
#           with no enclosing shard_map the W-term FFTs go through
#           ``make_sharded_*fftn_3d`` (which wraps its OWN shard_map) instead
#           of the interior ``local_*fftn3`` aliases -- i.e. this route is the
#           only one from which the flat-k FFT FFI is structurally reachable
#           (FFT_DONATION_AUDIT §3.1: shard_map cannot nest).
#
# The permanent hoist does not change the number of live
# (nk, mu_loc, nu_loc, ns^2)-class intermediates, which stays at the one
# ``T_b`` family documented below.
# ``krep`` REMOVED 2026-08-08 (owner ruling, measured): it uniquely removed 3
# in-loop collectives per block iteration that CGS2 does not, but they are a
# 128 B and two 13 KB all-reduces -- ~1.35 ms against a ~2700 ms eigensolve,
# 0.05%, and a 6-rep wall A/B could not see them in either direction inside a
# 30% intra-arm spread (FEAST_KPM_PASS.md §1).  It was never honoured on the
# shipped bs == 1 route, which is why it also carried an honesty banner and
# eight gate cells -- maintenance surface for a lever worth 0.05%.
# ``yhoist`` REMOVED 2026-08-08 -- not withdrawn, made PERMANENT.  It met the
# owner's standing knob policy exactly (improves performance, no drawback,
# arch-safe, so remove the knob and keep the winning behaviour): measured
# 1.007x alone -- inside the 3.42% run-to-run spread the same job measured by
# running its baseline cell twice -- at a cost of 16 KB/rank, with no scale
# cliff and no numerics (the gate asserted BIT-identity, not a tolerance).
# A lever that is free, harmless and too small to see is not a dial, it is a
# default.  ENV_KNOB_CENSUS.md §6.3, FIX_smallwins.md §3.5.
_MATVEC_OPTS = ("gspmd",)   # densek REMOVED 2026-07-31, owner directive


def matvec_opts() -> frozenset[str]:
    raw = os.environ.get("LORRAX_BSE_MATVEC_OPT", "").strip()
    if not raw:
        return frozenset()
    toks = [t.strip().lower() for t in raw.split(",") if t.strip()]
    bad = [t for t in toks if t not in _MATVEC_OPTS]
    if bad:
        raise ValueError(
            f"LORRAX_BSE_MATVEC_OPT={raw!r}: unknown option(s) {bad}.  "
            f"Valid tokens are {list(_MATVEC_OPTS)}, comma-separated; "
            f"unset/empty selects none.  Refusing rather than silently "
            f"running the baseline under an optimised label.")
    return frozenset(toks)



# ===========================================================================
# The three stages of the W term, factored so the A block and the coupling
# block share them.  These are the SAME einsums, in the same order, that
# ``_w_stack``'s body used inline before the port -- pure code motion, so the
# shipped TDA path is bit-identical (gated: test_bse_sp_lanczos.py::
# test_stack_matvec_tda_bit_identical_after_port).
# ===========================================================================

def _encode_T_A(X_b, psi_c_X, psi_v_Y):
    """A-block ISDF encode.  ``X_b`` (c_loc, v_full, nk) -> T (μ_loc,ν_loc,ns,ns,nk).

    ``T[μ,ν,t,s,k] = Σ_c ψ^X_c[k,c,t,μ] Σ_v conj(ψ^Y_v[k,v,s,ν]) X[c,v,k]``.
    μ rides 'x' (from ``psi_c_X``), ν rides 'y' (from ``psi_v_Y``).
    """
    R = jnp.einsum("kvsN,cvk->cksN", jnp.conj(psi_v_Y), X_b)   # (c_loc,nk,ns,ν_loc)
    Rc = lax.all_gather(R, "x", axis=0, tiled=True)            # (c_full,...)
    return jnp.einsum("kctM,cksN->MNtsk", psi_c_X, Rc)


def _encode_T_B(Xb_b, psi_c_Y, psi_v_X):
    """Coupling-block ISDF encode -- the c<->v leg swap (Henneke Eq. 4-3).

    ``T[μ,ν,t,s,k] = Σ_v ψ^X_v[k,v,t,μ] Σ_c conj(ψ^Y_c[k,c,s,ν]) X[c,v,k]``,
    ``Xb_b`` arriving as (c_full, v_loc, nk).  The legs swap but the SHARDING
    does not: μ still rides 'x' (now from ``psi_v_X``) and ν still rides 'y'
    (now from ``psi_c_Y``), so this T is add-compatible with ``_encode_T_A``'s
    with no collective and no resharding.  That is the fusion's precondition
    and it holds because LORRAX uses ONE ζ set for both legs (unlike Henneke's
    separate N_μ^vv / N_μ^cc / N_μ^vc).
    """
    R = jnp.einsum("kcsN,cvk->vksN", jnp.conj(psi_c_Y), Xb_b)  # (v_loc,nk,ns,ν_loc)
    Rv = lax.all_gather(R, "y", axis=0, tiled=True)            # (v_full,...)
    return jnp.einsum("kvtM,vksN->MNtsk", psi_v_X, Rv)


def _conv_decode(T_b, psi_c_X, psi_v_Y, W_R, nkx, nky, nkz, nk, sqrt_nk):
    """conv(T) then decode -- the eight stages both blocks share.

    conv: ``U_b = (1/√Nk) Σ_q W_q T_b[..., k−q]`` (ifft_k · W_R · fft_k).

    THIS IS AN FFT AND IT STAYS AN FFT.  A dense (nk x nk) DFT contraction gives
    the same numbers and measured 2.3x faster on this deck, and it was REMOVED
    on 2026-07-31 by owner directive: the dense form is O(nk^2) where the FFT is
    O(nk log nk), so it is a win only because nk = 16 here and it inverts at the
    thousand-k-point sizes LORRAX is being built for.  Do not reintroduce it,
    under any name, on any measurement.  If the k-transform is a bottleneck the
    answer is a better FFT (batching, an FFI handler, fewer dispatches), never a
    denser algorithm.

    The inverse transform is spelled conj(fft(conj(x))) so that BOTH transforms
    are forward and UNNORMALISED.  An ``ifft`` here would put the 1/nk inside
    XLA's FFT thunk, where it runs as a separate cuBLAS zscal over the whole
    T-tensor that no fusion pass can see -- a full extra HBM round trip per
    matvec.  The 1/nk is folded onto the decode output below instead.

    decode: ``(WX)_b = (1/√Nk) Σ_{μ,ν,t,s} conj(ψ_c) ψ_v U_b``.  psum_scatter
    completes the μ-sum while scattering c→x; the ν-sum's scatter to 'y' is
    hoisted OUT of the scan by the caller.
    """
    mu_loc, nu_loc, ns = T_b.shape[0], T_b.shape[1], T_b.shape[2]
    T_k = T_b.reshape(mu_loc, nu_loc, ns, ns, nkx, nky, nkz)
    T_R = jnp.conj(local_fftn3(jnp.conj(T_k), axes=(4, 5, 6), norm=None))
    U_R = W_R[:, :, None, None, :, :, :] * T_R
    U_b = local_fftn3(U_R, axes=(4, 5, 6), norm=None).reshape(
        mu_loc, nu_loc, ns, ns, nk)

    A = lax.psum_scatter(
        jnp.einsum("kctM,MNtsk->cNsk", jnp.conj(psi_c_X), U_b),
        "x", scatter_dimension=0, tiled=True)               # (c_loc, ν_loc, ns, nk)
    WXcv = jnp.einsum("kvsN,cNsk->cvk", psi_v_Y, A)         # (c_loc, v_full, nk)
    # Carries the two unnormalised transforms' 1/nk, on a (c_loc, v, nk) tensor
    # ~120x smaller than T that is being written anyway -- that is the whole
    # reason it is folded to here.
    return WXcv / (sqrt_nk * nk)


def build_bse_stack_matvec(
    mesh_xy: Mesh,
    nkx: int,
    nky: int,
    nkz: int,
    *,
    kernel: str = "bse",
):
    """Build the trial-stack BSE matvec.

    Parameters
    ----------
    kernel : {'bse', 'rpa'}
        ``'bse'`` returns ``D + V - W`` (screened direct term); ``'rpa'`` returns
        ``D + V`` (the W-term ``shard_map`` is not built).
    """
    if kernel not in ("rpa", "bse"):
        raise ValueError(f"kernel must be 'rpa' or 'bse', got {kernel!r}")
    include_W = kernel == "bse"

    sh = make_bse_shardings(mesh_xy)
    nk = nkx * nky * nkz
    opts = matvec_opts()
    use_gspmd = "gspmd" in opts

    # ── W term: one shard_map over ('x','y'); body = scan over the trial axis ──
    def _w_stack(X, psi_c_X, psi_v_Y, W_R):
        # Local shards: X (n_trials, c_loc, v_loc, nk); psi_c_X (nk, c_full, ns,
        # μ_loc); psi_v_Y (nk, v_full, ns, ν_loc); W_R (μ_loc, ν_loc, kx,ky,kz).
        # sqrt_nk follows the input dtype (fp32/fp64) — drop-in for fp32 GMRES.
        # DTYPE SEAM: this whole W-term inherits X's dtype, so a complex64 matvec
        # would halve the 655 MB T-tensor and every one of its ~7 HBM round-trips
        # (the audit's measured ~2× bandwidth lever, JOINT_FINDINGS §4). It is
        # DELIBERATELY left at complex128 (no c64 here) per owner decision
        # (2026-07-16); the fp32-GMRES path casts upstream in bse_feast, not here.
        sqrt_nk = jnp.sqrt(jnp.asarray(nk, dtype=X.real.dtype))

        def _body(carry, X_b):                       # X_b: (c_loc, v_full, nk)
            # encode: T_b[μ,ν,t,s,k] = Σ_c ψ_c[k,c,t,μ] Σ_v conj(ψ_v[k,v,s,ν]) X_b
            # The 'y' all-gather already happened OUTSIDE the scan, so X_b
            # arrives as (c_loc, v_full, nk) — one collective per BLOCK
            # instead of one per trial.  Encode / conv+decode are the shared
            # module-level stages (``_encode_T_A`` / ``_conv_decode``); the
            # coupling block reuses the SAME ``_conv_decode``, which is what
            # makes the non-TDA fusion exact.
            T_b = _encode_T_A(X_b, psi_c_X, psi_v_Y)
            # NB no per-trial psum_scatter on 'y' inside _conv_decode: it is
            # hoisted to one scatter for the whole block, after the scan.
            return carry, _conv_decode(T_b, psi_c_X, psi_v_Y, W_R,
                                       nkx, nky, nkz, nk, sqrt_nk)

        # ONE 'y' all-gather for the whole block instead of n_trials of them.
        # Operand (n_trials, c_loc, v_loc, nk) -> (…, v_full, …): 16 KB per
        # rank at P=64, against the 11.08 MB T_b the scan body already holds.
        # The scan carries no extra T-class buffer.
        X = lax.all_gather(X, "y", axis=2, tiled=True)
        _, WX = lax.scan(_body, None, X)             # WX: (n_trials, c_loc, v_*, nk)
        WX = lax.psum_scatter(WX, "y", scatter_dimension=2, tiled=True)
        return WX

    # ── W term, GSPMD twin: same math, same scan, NO shard_map ────────────────
    # Audit route (``LORRAX_BSE_MATVEC_OPT=gspmd``).  Line-for-line the same
    # chain as ``_w_stack`` above, but on GLOBAL arrays.  Each of the four
    # ``with_sharding_constraint`` calls below sits at exactly the point where
    # the manual body issues a collective, and requests the SAME data layout the
    # manual collective produces -- so if the partitioner is any good it should
    # emit the same four collectives.  Whether it does is the experiment.
    #
    #   manual                                    | gspmd hint
    #   all_gather(X_b, 'y', axis=1)              | wsc(X_b,  P('x', None, None))
    #   all_gather(R,   'x', axis=0)              | wsc(R,    P(None, None, None, 'y'))
    #   psum_scatter(..., 'x', scatter_dim=0)     | wsc(A,    P('x', 'y', None, None))
    #   psum_scatter(..., 'y', scatter_dim=1)     | wsc(WXcv, P('x', 'y', None))
    #
    # The FFTs necessarily change door: with no enclosing shard_map the interior
    # ``local_*fftn3`` aliases would gather the (μ,ν)-sharded operand onto every
    # rank (fft_helpers:221-226 forbids exactly that), so this route uses the
    # ``make_sharded_*fftn_3d`` factories, which wrap the identical local kernel
    # in their own shard_map.  That shard_map is NOT nested here -- which is the
    # structural point this route exists to demonstrate.
    _ns = lambda spec: NamedSharding(mesh_xy, spec)
    _T7_spec = P("x", "y", None, None, None, None, None)
    _g_ifftn = make_sharded_ifftn_3d(
        mesh_xy, _T7_spec, _T7_spec, axes=(4, 5, 6), norm="ortho")
    _g_fftn = make_sharded_fftn_3d(
        mesh_xy, _T7_spec, _T7_spec, axes=(4, 5, 6), norm="ortho")

    def _w_gspmd(X, psi_c_X, psi_v_Y, W_R):
        # Global shapes: X (n_trials, c, v, nk); psi_c_X (nk, c, ns, μ);
        # psi_v_Y (nk, v, ns, ν); W_R (μ, ν, kx, ky, kz).
        sqrt_nk = jnp.sqrt(jnp.asarray(nk, dtype=X.real.dtype))

        def _body(carry, X_b):                       # X_b: (c, v, nk) global
            # 'y' gather: make v replicated so the ν-contraction below is local.
            Xv = lax.with_sharding_constraint(X_b, _ns(P("x", None, None)))
            R = jnp.einsum("kvsN,cvk->cksN", jnp.conj(psi_v_Y), Xv)
            # 'x' gather: make c replicated so the μ-encode below is local.
            R = lax.with_sharding_constraint(R, _ns(P(None, None, None, "y")))
            T_b = jnp.einsum("kctM,cksN->MNtsk", psi_c_X, R)
            T_b = lax.with_sharding_constraint(
                T_b, _ns(P("x", "y", None, None, None)))
            mu, nu, ns = T_b.shape[0], T_b.shape[1], T_b.shape[2]

            T_k = T_b.reshape(mu, nu, ns, ns, nkx, nky, nkz)
            T_R = _g_ifftn(T_k)
            U_R = W_R[:, :, None, None, :, :, :] * T_R
            U_b = _g_fftn(U_R).reshape(mu, nu, ns, ns, nk)

            # μ-sum with c landing on 'x' -- the psum_scatter('x') ask.
            A = jnp.einsum("kctM,MNtsk->cNsk", jnp.conj(psi_c_X), U_b)
            A = lax.with_sharding_constraint(A, _ns(P("x", "y", None, None)))
            # ν-sum with v landing on 'y' -- the psum_scatter('y') ask.
            WXcv = jnp.einsum("kvsN,cNsk->cvk", psi_v_Y, A)
            WXcv = lax.with_sharding_constraint(WXcv, _ns(P("x", "y", None)))
            return carry, WXcv / sqrt_nk

        _, WX = lax.scan(_body, None, X)
        return lax.with_sharding_constraint(WX, sh.X)

    w_stack = _shard_map_fn(
        _w_stack,
        mesh=mesh_xy,
        in_specs=(P(None, "x", "y", None), P(None, None, None, "x"),
                  P(None, None, None, "y"), P("x", "y", None, None, None)),
        out_specs=P(None, "x", "y", None),
    )
    if use_gspmd:
        w_stack = _w_gspmd

    def _matvec(X, psi_c_X, psi_c_Y, psi_v_X, psi_v_Y, eps_c, eps_v, W_R, V_q0,
                M_X, M_Y):
        # M_X (μ on x) / M_Y (ν on y): hoisted exchange pair amplitudes, precomputed
        # once per solve (audit P3). psi_c_Y / psi_v_X are now unused here — kept for
        # a uniform matvec signature with the ring paths; psi_c_X / psi_v_Y still
        # feed the W-term.
        sqrt_nk = jnp.sqrt(jnp.asarray(nk, dtype=X.real.dtype))
        # ── D term: (ε_c − ε_v) · X  (batched, local) ──────────────────────────
        delta_E = eps_c.T[None, :, None, :] - eps_v.T[None, None, :, :]
        D_term = lax.with_sharding_constraint(delta_E * X, sh.X)

        # ── V term: B1 dense exchange, k-summed encode + broadcast decode ──────
        # K^x = M V M†: conjugated vertex on the encode leg, bare vertex on the
        # decode.  Fixed by the transition density <0|ρ̂|Ψ> = Σ A_cvk ψ_ck ψ*_vk;
        # the reverse assignment builds conj(K^x), which cannot be covariant
        # alongside the (correct, untouched) W term.
        S = jnp.einsum("kcvN,bcvk->bN", jnp.conj(M_Y), X)         # k SUMMED → (b, ν_loc)
        S = lax.with_sharding_constraint(S, sh.S_k0) / sqrt_nk
        U = jnp.einsum("MN,bN->bM", V_q0, S)                      # (b, μ_loc)
        U = lax.with_sharding_constraint(U, sh.d_mu)
        VX = jnp.einsum("kcvM,bM->bcvk", M_X, U)                 # broadcast over k
        VX = lax.with_sharding_constraint(VX, sh.X) / sqrt_nk

        if not include_W:
            return D_term + VX

        WX = w_stack(X, psi_c_X, psi_v_Y, W_R)
        return D_term + VX - WX

    return jax.jit(
        _matvec,
        in_shardings=(sh.X, sh.psi_x, sh.psi_y, sh.psi_x, sh.psi_y,
                      sh.eps, sh.eps, sh.W, sh.V, sh.psi_x, sh.psi_y),
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

    # ── W term: one shard_map over ('x','y') — the SAME single region the TDA
    #    stack matvec opens.  No new shard_map is created by the coupling port:
    #    the B encode is an einsum pair plus one all_gather INSIDE this body.
    def _w_pair(X, Xb, sc, psi_c_X, psi_c_Y, psi_v_X, psi_v_Y, W_R):
        # Local shards: X, Xb (n_trials, c_loc, v_loc, nk); psi_*_X (…, μ_loc);
        # psi_*_Y (…, ν_loc); W_R (μ_loc, ν_loc, kx, ky, kz).
        sqrt_nk = jnp.sqrt(jnp.asarray(nk, dtype=X.real.dtype))

        # Two block-level gathers, both hoisted OUT of the scan.  The A encode
        # wants v replicated; the B encode wants c replicated.  Both operands
        # are (n_trials, c_loc, v_loc, nk) — 16 KB per rank at P=64 — against
        # the 11 MB T the scan body already holds, so this is the same trade
        # the permanent 'y' hoist already makes, taken twice.
        X_y = lax.all_gather(X, "y", axis=2, tiled=True)    # (b, c_loc, v_full, nk)
        Xb_x = lax.all_gather(Xb, "x", axis=1, tiled=True)  # (b, c_full, v_loc, nk)

        def _body_fused(carry, xs):
            X_b, Xb_b = xs
            # THE FUSION.  T^A and T^B have identical shape AND identical
            # sharding, so this add needs no collective and no reshard; conv
            # and decode are linear, so one chain serves both blocks.
            T_b = (_encode_T_A(X_b, psi_c_X, psi_v_Y)
                   + sc * _encode_T_B(Xb_b, psi_c_Y, psi_v_X))
            return carry, _conv_decode(T_b, psi_c_X, psi_v_Y, W_R,
                                       nkx, nky, nkz, nk, sqrt_nk)

        def _body_unfused(carry, xs):
            # THE TWIN.  Two full chains.  Kept only to price the fusion.
            X_b, Xb_b = xs
            WA = _conv_decode(_encode_T_A(X_b, psi_c_X, psi_v_Y),
                              psi_c_X, psi_v_Y, W_R, nkx, nky, nkz, nk, sqrt_nk)
            WB = _conv_decode(_encode_T_B(Xb_b, psi_c_Y, psi_v_X),
                              psi_c_X, psi_v_Y, W_R, nkx, nky, nkz, nk, sqrt_nk)
            return carry, WA + sc * WB

        _, WX = lax.scan(_body_fused if fuse else _body_unfused,
                         None, (X_y, Xb_x))
        return lax.psum_scatter(WX, "y", scatter_dimension=2, tiled=True)

    w_pair = _shard_map_fn(
        _w_pair,
        mesh=mesh_xy,
        in_specs=(P(None, "x", "y", None), P(None, "x", "y", None), P(),
                  P(None, None, None, "x"), P(None, None, None, "y"),
                  P(None, None, None, "x"), P(None, None, None, "y"),
                  P("x", "y", None, None, None)),
        out_specs=P(None, "x", "y", None),
    )

    def _pair(X, s, psi_c_X, psi_c_Y, psi_v_X, psi_v_Y, eps_c, eps_v, W_R,
              V_q0, M_X, M_Y):
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
        S_A = jnp.einsum("kcvN,bcvk->bN", jnp.conj(M_Y), X)       # (b, ν_loc)
        S_B = jnp.einsum("kcvN,bcvk->bN", M_Y, Xb)                # (b, ν_loc)
        S = lax.with_sharding_constraint(S_A + sc * S_B, sh.S_k0) / sqrt_nk
        U = jnp.einsum("MN,bN->bM", V_q0, S)                      # (b, μ_loc)
        U = lax.with_sharding_constraint(U, sh.d_mu)
        VX = jnp.einsum("kcvM,bM->bcvk", M_X, U)                  # broadcast over k
        VX = lax.with_sharding_constraint(VX, sh.X) / sqrt_nk

        if not include_W:
            return D_term + VX

        WX = w_pair(X, Xb, sc, psi_c_X, psi_c_Y, psi_v_X, psi_v_Y, W_R)
        return D_term + VX - WX

    return jax.jit(
        _pair,
        in_shardings=(sh.X, rep, sh.psi_x, sh.psi_y, sh.psi_x, sh.psi_y,
                      sh.eps, sh.eps, sh.W, sh.V, sh.psi_x, sh.psi_y),
        out_shardings=sh.X,
    )
