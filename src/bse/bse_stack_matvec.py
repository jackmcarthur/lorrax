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

Retirement plan (PARTIALLY EXECUTED, 2026-08-08) — the ring/gather/simple TDA
matvecs existed only to bound ``T``'s peak; this scan bounds it strictly better,
so they are superseded.  Consumers repointed here: ``bse_lanczos.solve_bse_sharded``
(block-Lanczos + Davidson) and ``bse_feast`` (TDA GMRES contour solves +
``_rayleigh_ritz`` subspace application).  What the plan asked for, and where it
now stands:
  * DONE — the non-TDA ``S=[[A,B],[-B†,-A†]]`` builder now lives here too.
    ``bse_ring_comm.build_bse_ring_matvec_full`` is a compatibility adapter;
    its former low-memory and gather routes are branches of this shared stack
    builder and remain value-gated against each other.
  * DONE — the ``krep`` matvec option and the bare-``shard_map`` sites are
    deleted, and the ``yhoist`` collective hoist is unconditional (``3a7704bb``,
    ``8349b65c``, ``ac67fd3c``).  ``LORRAX_BSE_MATVEC_OPT`` survives because
    ``gspmd`` remains — see the dial's own note below.
  * STILL OPEN — ``bse_ring_comm.build_bse_ring_matvec`` (TDA), used by
    ``bse_feast.estimate_spectral_bounds_sharded`` (spectral-bound Lanczos) and
    the equality gates.  Repoint + delete together with ``bse_simple`` and the
    ``matvec_kind`` CLI flag/data key once the spectral-bound Lanczos is moved
    over.  The flag is ALREADY INERT on the sharded eigensolve path
    (``bse_lanczos.py``, "the legacy ``matvec_kind`` selector is retired here"):
    it is still parsed and still steers ``absorption_haydock``, so deleting it
    is a driver-surface change, not a rename.  This is the one row of the plan
    that needs an owner call rather than a follow-up commit.

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
from common.contract_bands import contract_bands_block_reshard

from common.fft_helpers import (
    local_ifftn3,
    local_fftn3,
    make_sharded_fftn_3d,
    make_sharded_ifftn_3d,
)
from .bse_ring_comm import (
    _ring_sum_B_encode,
    _ring_sum_conduction,
    _ring_sum_valence,
    apply_V_ring,
    make_bse_shardings,
    ring_spin_degeneracy,
)
from .w_ladder_conv_kminor import build_rung_body, rung_uses_conv_kminor


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
#   [densek REMOVED 2026-07-31 — owner directive, no exceptions.  The ruling
#           and the numbers behind it are stated once, at the site it governs:
#           ``_conv_decode``'s "THIS IS AN FFT AND IT STAYS AN FFT".]
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
    ``Xb_b`` arrives as (c_full, v_loc, nk) and is gathered to
    (c_full, v_full, nk) here, so BOTH ζ legs stay stationary.  The legs
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
    Xb_full = lax.all_gather(Xb_b, "y", axis=1, tiled=True)    # (c_full,v_full,nk)
    R = jnp.einsum("kcsN,cvk->vksN", jnp.conj(psi_c_Y), Xb_full)  # (v_full,nk,ns,ν_loc)
    return jnp.einsum("kvtM,vksN->MNtsk", psi_v_X, R)


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
    head_tensor: bool = False,
    full: bool = False,
    low_mem: bool = True,
    screening: bool = False,
    return_half_appliers: bool = False,
    ladder_rung_slots: bool = False,
    fuse_ladder_rung: bool = True,
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

        is rank three over transitions and belongs beside ``M_X``/``M_Y``,
        where this matvec already carries rank-three objects
        (``LT_HEAD_PROBLEM.md`` §6).

        Structurally it IS the exchange term with ``(M_X, M_Y, V_q0)``
        replaced by ``(D_head, D_head, M_head)``, which is why it reuses the
        same encode/decode shape and the same ``1/N_k``.  ``D_head`` is
        ``conj(d)`` with the same ``(k, c, v, a)`` layout as ``M_X`` and a
        Cartesian axis of length 3 where μ was; ``M_head`` is the real
        symmetric ``(3, 3)`` cell moment.  Hermiticity of the added term is
        then automatic: ``M`` real symmetric ⇒ ``K^head`` Hermitian.
    full : bool
        Build the non-TDA pair-space operator. The remaining options below
        are legal only on this route.
    low_mem : bool
        Select ring communication (``True``) or the all-gather encode twin.
    screening : bool
        Select the density-response block convention rather than the optical
        BSE convention.
    return_half_appliers, ladder_rung_slots, fuse_ladder_rung : bool
        Full-operator interfaces retained for dense construction and the
        ladder resolvent; their detailed contracts live on the shared full
        implementation below.
    """
    if kernel not in ("rpa", "bse"):
        raise ValueError(f"kernel must be 'rpa' or 'bse', got {kernel!r}")
    if full:
        if head_tensor:
            raise ValueError("head_tensor=True is not implemented for full=True")
        return _build_bse_stack_matvec_full(
            mesh_xy, nkx, nky, nkz, low_mem=low_mem,
            include_W=kernel == "bse", screening=screening,
            return_half_appliers=return_half_appliers,
            ladder_rung_slots=ladder_rung_slots,
            fuse_ladder_rung=fuse_ladder_rung,
        )
    if (not low_mem or screening or return_half_appliers
            or ladder_rung_slots or not fuse_ladder_rung):
        raise ValueError(
            "low_mem, screening, return_half_appliers, ladder_rung_slots, "
            "and fuse_ladder_rung are full=True options")
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
                M_X, M_Y, D_head=None, M_head=None):
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
        spin_deg = ring_spin_degeneracy(psi_c_X.shape[2])
        S = (lax.with_sharding_constraint(S, sh.S_k0) / sqrt_nk) * spin_deg
        U = jnp.einsum("MN,bN->bM", V_q0, S)                      # (b, μ_loc)
        U = lax.with_sharding_constraint(U, sh.d_mu)
        VX = jnp.einsum("kcvM,bM->bcvk", M_X, U)                 # broadcast over k
        VX = lax.with_sharding_constraint(VX, sh.X) / sqrt_nk

        if head_tensor:
            # ── Head term: the SAME contraction with (D_head, M_head) in
            #    place of (M_Y, V_q0, M_X).  Three Cartesian components stand
            #    where μ stood, so this is three inner products per trial
            #    vector and a 3x3 — free next to everything above.  The
            #    conjugation follows the V term's exactly, and must: the
            #    encode leg carries the conjugated vertex.
            # D_head = conj(d), so conj(D_head) is the bare dipole and the
            # two legs read exactly as M_Y / M_X do above.
            Sh = (jnp.einsum("kcva,bcvk->ba", jnp.conj(D_head), X)
                  / sqrt_nk) * spin_deg
            Uh = Sh @ M_head.astype(Sh.dtype).T                   # U_a = M_ab S_b
            HX = jnp.einsum("kcva,ba->bcvk", D_head, Uh)
            VX = VX + lax.with_sharding_constraint(HX, sh.X) / sqrt_nk

        if not include_W:
            return D_term + VX

        WX = w_stack(X, psi_c_X, psi_v_Y, W_R)
        return D_term + VX - WX

    in_sh = [sh.X, sh.psi_x, sh.psi_y, sh.psi_x, sh.psi_y,
             sh.eps, sh.eps, sh.W, sh.V, sh.psi_x, sh.psi_y]
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
        S = (lax.with_sharding_constraint(S_A + sc * S_B, sh.S_k0)
             / sqrt_nk) * ring_spin_degeneracy(psi_c_X.shape[2])
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


def _build_bse_stack_matvec_full(
    mesh_xy: Mesh,
    nkx: int,
    nky: int,
    nkz: int,
    low_mem: bool = True,
    include_W: bool = True,
    screening: bool = False,
    return_half_appliers: bool = False,
    ladder_rung_slots: bool = False,
    fuse_ladder_rung: bool = True,
):
    """Build full (non-TDA) BSE matvec ``[X;Y] -> H[X;Y]``.

    ``fuse_ladder_rung`` (LADDER SCREENING ONLY — ``screening and include_W``;
    inert everywhere else) sums the two rung ``T`` intermediates of each block
    row before the FFT chain rather than after, so the matvec runs the
    ``(nb, mu, nu, s, s, k)`` chain TWICE instead of four times.  Same operator,
    same conjugation convention, different association order (so agreement with
    the unfused core is to rounding, not to the bit).  ``False`` selects the
    unfused core for the A/B and for bisection — see ``_impl_core_fused``.

    ``screening`` selects the coupling-block kernel AND the anti-resonant row
    (``_antiresonant_row``), which together fix *which* physical operator this is:

    - ``screening=False`` (default): the OPTICAL BSE, the para-Hermitian
      ``H = [[A, B], [-B*, -A*]]`` (* = complex conjugate) with ``A`` Hermitian
      and ``B`` complex-SYMMETRIC.  The B (coupling) block uses the excitonic
      exchange ``V_B`` (Henneke Eq. 2-20, conjugated pairing
      ``⟨M_t|v|conj(M_t')⟩`` — ``apply_V_ring_B``) plus the c'↔v'-swapped direct
      term.  With ``include_W`` this is the TDHF/BSE exciton Hamiltonian; its
      spectrum is REAL with +-omega pairs (solved by ``bse_nontda``).  NOTE: the
      historical ``-B, -A`` anti-resonant row gave a COMPLEX spectrum for complex
      B and was never value-validated — fixed here (PHASE2_LOG "non-TDA
      eigensolvers").
    - ``screening=True``: the test-charge DENSITY response (the object whose
      resolvent gives the screened Coulomb ``W = v + vχv``).  Here the B block
      uses the SAME RING kernel ``K^A = (1/Nk)⟨M_t|v|M_t'⟩`` as the A block
      (``apply_V_ring``), and the ring part of the anti-resonant row is
      UN-conjugated.  Both facts follow from the density-vertex convention that
      makes this operator a density response at all (derivation:
      ``w_ladder`` module docstring, "the Hartree rung is a dyad"): the ring
      part of ``H`` is the outer product of the seed injection ``[v; -v]`` and
      the density readout ``X + Y``, so it is branch-blind and identical in all
      four blocks.  See ``bse_w_exact`` for the identity
      ``W(0) − v = v(0 − H)⁻¹v`` and its per-element convention.

      ``include_W`` selects WHICH screening diagram set the response resums:

      * ``include_W=False`` — RPA: ``A = D + V``, ``B = V`` ⇒ ``A − B = D``,
        ``χ = χ₀(1 − vχ₀)⁻¹``.  This is the W that GW's Dyson solve produces
        and the ONLY legal choice while W itself is being built.
      * ``include_W=True`` — LADDER (``W_BSE``): the statically screened direct
        rung ``−W`` is added to the irreducible part, ``A = D + V − W_d`` and
        ``B = V − W_d^B``, so ``χ = P̃(1 − vP̃)⁻¹`` with ``P̃`` the
        ladder-corrected irreducible polarizability.  The direct terms are the
        OPTICAL ones verbatim (``_apply_A``'s ``W_term``, ``_apply_B``'s
        c'↔v'-swapped ``W_term``) — the direct rung contracts band pairs
        through W and never touches a density vertex, so the screening /
        optical convention difference cannot reach it — but the anti-resonant
        row CONJUGATES them while leaving the ring term alone (see
        ``_antiresonant_row``; measured, not assumed).  Requires a W_R built
        from an already-converged W(0) (``ensure_W_R(..., include_W=True)``),
        i.e. the two-stage structure of the feature.  Neither ``A − B = D`` nor
        the plain symplectic ``[[A,B],[-B,-A]]`` form survives, which is why
        ``w_omega_chain``'s z² reduction is refused for this operator (see
        ``w_ladder.refuse_chain_path``).

      ``ladder_rung_slots`` declares the PAYLOAD convention: the finite-q
      payload (``bse_w_exact.build_finite_q_data``) stores ``conj(psi)`` on
      both legs — exact for the four density vertices, but the direct rung is
      bilinear in (c, c') and (v, v') band pairs and must consume the
      PHYSICAL (rolled, un-flipped) arrays.  With the flag set, the matvec
      signature grows four trailing psi operands for the rung
      (``bse_feast.ladder_matvec_operands``; ``build_finite_q_data`` supplies
      them).  Left unfixed, the rung ran on the conjugated arrays — a defect
      value-invisible at q=0 and measured as a 3.6e-4 violation of
      ``W(-q) = conj(W(q))`` at finite q (claim 0215, 2026-08-15).  A
      conj-wrap compensation inside the rung appliers was tried first and
      REFUTED by block-level measurement (probe_block_compare, 2026-08-16:
      ~7e-5 residual against the dense operator; physical operands measure
      1e-20) — the physical arrays are supplied, not reconstructed.
    """
    if ladder_rung_slots and not (screening and include_W):
        raise ValueError(
            "ladder_rung_slots=True is only meaningful for the ladder "
            "screening operator (screening=True, include_W=True): it extends "
            "the matvec signature with the four PHYSICAL (rolled, un-flipped) "
            "psi operands the direct rung consumes on a build_finite_q_data "
            "payload, and no other operator both takes that payload and "
            "carries a rung. Got screening=%r, include_W=%r." % (screening, include_W))
    if ladder_rung_slots and return_half_appliers:
        raise ValueError(
            "ladder_rung_slots does not compose with return_half_appliers: "
            "no half-applier caller exists for the ladder operator (the "
            "chain path is refused, w_ladder.refuse_chain_path).")
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
        v_chunk = X.shape[2]
        n_rmu_local_X = psi_v_X.shape[-1]
        n_rmu_local_Y = psi_c_Y.shape[-1]
        return _ring_sum_B_encode(X, psi_c_Y, psi_v_X, v_chunk, px, py,
                                  n_rmu_local_X, n_rmu_local_Y)

    encode_T_ring_B = _shard_map_fn(
        _encode_T_B,
        mesh=mesh_xy,
        in_specs=(P(None, "x", "y", None), P(None, None, None, "y"), P(None, None, None, "x")),
        out_specs=P(None, "x", "y", None, None, None),
    )

    def _encode_T_B_gather(X, psi_c_Y, psi_v_X):
        # Gather the TRIAL VECTOR on both axes, never the partially-contracted
        # R: R carries nu on 'y', and an all_gather of R along v concatenates
        # tiles whose nu shards differ -- the same defect as the ring version
        # (see _ring_sum_B_encode).  X carries no zeta axis, so gathering it is
        # sound, and it is the smallest tensor in the chain.
        X_full_c = lax.all_gather(X, "x", axis=1, tiled=True)
        X_full = lax.all_gather(X_full_c, "y", axis=2, tiled=True)
        R = jnp.einsum("kcsN,bcvk->bvksN", jnp.conj(psi_c_Y), X_full)
        T = jnp.einsum("kvtM,bvksN->bMNtsk", psi_v_X, R)
        return T

    encode_T_gather_B = _shard_map_fn(
        _encode_T_B_gather,
        mesh=mesh_xy,
        in_specs=(P(None, "x", "y", None), P(None, None, None, "y"), P(None, None, None, "x")),
        out_specs=P(None, "x", "y", None, None, None),
    )

    def _apply_V_ring_only(X, psi_c_Y, psi_v_Y, M_X, V_q0):
        return apply_V_ring(X, psi_c_Y, psi_v_Y, M_X, V_q0, nk, px, py)

    apply_V_ring_only = _shard_map_fn(
        _apply_V_ring_only,
        mesh=mesh_xy,
        in_specs=(P(None, "x", "y", None), P(None, None, None, "y"), P(None, None, None, "y"),
                  P(None, None, None, "x"), P("x", "y")),
        out_specs=P(None, "x", "y", None),
    )

    def _apply_V_ring_B(X, psi_c_Y, psi_v_Y, M_X, V_q0):
        return apply_V_ring(
            X,
            jnp.conj(psi_c_Y),
            jnp.conj(psi_v_Y),
            M_X,
            V_q0,
            nk,
            px,
            py,
        )

    apply_V_ring_B = _shard_map_fn(
        _apply_V_ring_B,
        mesh=mesh_xy,
        in_specs=(P(None, "x", "y", None), P(None, None, None, "y"), P(None, None, None, "y"),
                  P(None, None, None, "x"), P("x", "y")),
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

    # ψ†Uψ decode = contract_bands_block_reshard, extra="leading" (owner
    # order 2026-07-29; adoption map wk_REL/contract_bands_notes.md §6.2 —
    # the CLEAN drop-in site: the b-stacked U already exists, so the stack
    # axis is free).  Replaces the partitioner-chosen collectives of the
    # historical einsum pair ("kctM,bMNtsk->bcNsk" then "kvsN,bcNsk->bcvk",
    # c-replicated intermediate, LARGE payload on the strided 'x' groups —
    # the exact inversion the primitive's §3.2 policy refuses) with the
    # structural stacked psum_scatter chain: large partial over the
    # node-local 'y' groups, small final over 'x', all b trials on ONE
    # collective per mesh axis (AK.9), impl=mpi warm-up inherited from the
    # factory.  Value-level identical (contraction reassociation — gate at
    # 1e-12, not bit-exact).  The transposes below are rank-local
    # (sharded axes preserved: M stays on 'x', N on 'y', c on 'x', v on
    # 'y'); the U transpose to the primitive's k-leading layout is priced
    # by the parity/perf gate, and composes with the future flat-k conv
    # layout (map §6.1 route (a)) which emits k-leading natively.
    _w_decode = contract_bands_block_reshard(
        mesh_xy, extra="leading",
        divisibility_hint=(
            "BSE callers: n_cond_pad / n_val_pad already pad c to p_x and "
            "v to p_y (bse_io loader); an indivisible window here means an "
            "unpadded/hand-built operand."))

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

        # (b, M, N, t, s, k) -> (b, k, t, M, s, N): the primitive's
        # canonical O layout (extra="leading"); ψ_v (k, v, s, N) ->
        # (k, s, N, v) = ψ_right.  conj(ψ_c) is applied inside.
        O_b = jnp.transpose(U, (0, 5, 3, 1, 4, 2))
        psi_v_snv = jnp.transpose(psi_v_Y, (0, 2, 3, 1))
        out = _w_decode(psi_c_X, O_b, psi_v_snv)     # (b, nk, c_X, v_Y)
        WX = jnp.transpose(out, (0, 2, 3, 1))        # (b, c_X, v_Y, nk)
        return WX / sqrt_nk

    # --- the fused-conv family's k-MINOR member, DIAL `auto` --------------
    # LORRAX_CONV_KMINOR_FFI=auto (the default) replaces the chain above —
    # reshape / ifftn / W_R multiply / fftn / reshape / transpose-to-O — with
    # ONE FFI call that also emits the decode's O layout from its store,
    # WHEN the mesh is CUDA, the device library exports the handler and the
    # k-grid's row is shared-memory resident.  Otherwise this line is a no-op
    # and the body above runs unchanged, which is the certified path on every
    # backend.  Same signature, same operands, same output sharding; measured
    # rel <= 6e-16 against this body and gated in
    # tests/bench/bench_conv_kminor.py.  Everything behind the dial lives in
    # bse.w_ladder_conv_kminor, so this file keeps exactly ONE spelling of the
    # chain plus this hook.
    _ck_use, _ck_why = rung_uses_conv_kminor(mesh_xy, (nkx, nky, nkz),
                                             jnp.complex128)
    if _ck_use:
        _apply_W_from_T = build_rung_body(mesh_xy, (nkx, nky, nkz), _w_decode)

    apply_W_from_T = jax.jit(
        _apply_W_from_T,
        in_shardings=(sh.T, sh.psi_x, sh.psi_y, sh.W),
        out_shardings=sh.X,
        # NB: T (arg 0) is NOT donated — the WX output has a different shape
        # (nt,c,v,k) so the donation is always declined (no aliasable output) and
        # emits no fallback copy. Dropping the cosmetic donate_argnums silences the
        # recurring "donated buffers not usable" warning (audit P5, JOINT_FINDINGS §3).
    )

    def _apply_D_term(X, eps_c, eps_v):
        delta_E = eps_c.T[None, :, None, :] - eps_v.T[None, None, :, :]
        return delta_E * X

    apply_D_term = jax.jit(
        _apply_D_term,
        in_shardings=(sh.X, sh.eps, sh.eps),
        out_shardings=sh.X,
    )

    def _T_term_A(X, psi_cW_X, psi_vW_Y):
        """The rung's ``(mu, nu, s, s, k)`` intermediate for ``W_d X``.

        Split out of :func:`_w_term_A` so a caller that applies SEVERAL rung
        terms with the same ``(psi, W_R)`` can sum their ``T`` first and run the
        FFT chain once — see ``_impl_core``'s fused row.  No behaviour of its
        own: ``_w_term_A`` is still exactly ``apply_W_from_T(_T_term_A(...))``.
        """
        return (encode_T_ring(X, psi_cW_X, psi_vW_Y) if low_mem
                else encode_T_gather(X, psi_cW_X, psi_vW_Y))

    def _T_term_B(X, psi_cW_Y, psi_vW_X):
        """The same, for the c'<->v'-swapped (coupling) rung ``W_d^B X``."""
        return (encode_T_ring_B(X, psi_cW_Y, psi_vW_X) if low_mem
                else encode_T_gather_B(X, psi_cW_Y, psi_vW_X))

    def _w_term_A(X, psi_cW_X, psi_vW_Y, W_R):
        """``W_d X`` — the resonant direct (screened-exchange) rung alone.

        The psi operands here are the RUNG's own: on a raw payload they are
        the density-vertex arrays; on a ``build_finite_q_data`` payload they
        MUST be the rolled UN-flipped arrays (``ladder_rung_slots``).  The
        conjugated-psi density convention is exact for the four density
        vertices and WRONG for this bilinear band-pair rung — a conj-wrap
        compensation (``conj(rung(conj X))``, valid-looking by an elementwise
        argument) was tried first and REFUTED by block-level measurement
        (probe_block_compare, 2026-08-16: ~7e-5 against the dense operator,
        vs 1e-20 with physical operands)."""
        return apply_W_from_T(_T_term_A(X, psi_cW_X, psi_vW_Y),
                             psi_cW_X, psi_vW_Y, W_R)

    def _w_term_B(X, psi_cW_X, psi_cW_Y, psi_vW_X, psi_vW_Y, W_R):
        """``W_d^B X`` — the c'<->v'-swapped (coupling) direct rung alone.

        Same operand contract as ``_w_term_A``."""
        return apply_W_from_T(_T_term_B(X, psi_cW_Y, psi_vW_X),
                             psi_cW_X, psi_vW_Y, W_R)

    def _apply_A(X, psi_c_X, psi_c_Y, psi_v_X, psi_v_Y, eps_c, eps_v, W_R, V_q0,
                 M_X, psi_cW_X, psi_vW_Y):
        D_term = apply_D_term(X, eps_c, eps_v)
        V_term = apply_V_ring_only(X, psi_c_Y, psi_v_Y, M_X, V_q0)
        if not include_W:
            return D_term + V_term
        return D_term + V_term - _w_term_A(X, psi_cW_X, psi_vW_Y, W_R)

    def _apply_B(X, psi_c_X, psi_c_Y, psi_v_X, psi_v_Y, W_R, V_q0, M_X,
                 psi_cW_X, psi_cW_Y, psi_vW_X, psi_vW_Y):
        # screening (RPA density response): ring kernel K^A, same as the A block
        # (apply_V_ring_only); optical BSE: excitonic V_B (apply_V_ring_B). Both take
        # the SAME hoisted M_X (audit P3) — apply_V_ring_B conjugates only ψ^Y.
        if screening:
            V_term = apply_V_ring_only(X, psi_c_Y, psi_v_Y, M_X, V_q0)
        else:
            V_term = apply_V_ring_B(X, psi_c_Y, psi_v_Y, M_X, V_q0)
        if not include_W:
            return V_term
        return V_term - _w_term_B(X, psi_cW_X, psi_cW_Y, psi_vW_X, psi_vW_Y, W_R)

    def _antiresonant_row(X, Y, psi_c_X, psi_c_Y, psi_v_X, psi_v_Y, eps_c, eps_v,
                          W_R, V_q0, M_X,
                          psi_cW_X, psi_cW_Y, psi_vW_X, psi_vW_Y):
        """Bottom (anti-resonant) block-row of the non-TDA operator applied to
        ``[X; Y]``: ``Y_out``.  The physics of this row depends on ``screening``:

        * ``screening=True, include_W=False`` (RPA test-charge density response):
          the coupling ``B = K^A`` is Hermitian and ``A`` is Hermitian, so the
          operator is the symplectic ``[[A, B], [-B, -A]]`` and
          ``Y_out = -B X - A Y``.  Validated by the W(0) resolvent closure
          (PHASE2_LOG "W(0)").

        * ``screening=True, include_W=True`` (LADDER density response): the row
          is a HYBRID, and the split is not cosmetic.

            Y_out = -[V X - conj(W_d^B conj(X))] - [D Y + V Y - conj(W_d conj(Y))]

          The RING term keeps the RPA row's sign and NO conjugate, because the
          Hartree rung is the dyad ``[M; -M] v [M^dag, M^dag]`` whose bottom row
          IS ``-K^A`` — that is what makes ``W - v = v chi v`` hold at all (the
          Woodbury identity in the ``w_ladder`` module docstring).  The DIRECT
          terms are CONJUGATED, because the ladder rung is an ordinary
          two-particle kernel whose anti-resonant blocks are the complex
          conjugates of the resonant ones (``K^AA = conj(K^RR)``,
          ``K^AR = conj(K^RA) = (K^RA)^dag`` since ``W_d^B`` is complex
          SYMMETRIC) — the same para-Hermitian structure the optical row uses,
          and the reason the ladder-only spectrum here IS the exchange-free
          optical BSE's.

          MEASURED, on the gnppm 2v2c fixture at q=0 (probe leg, JID 57052808):
          the naive un-conjugated row ``-B X - A Y`` leaves the static tile
          NON-Hermitian at ``max|W - W^dag|/|W| = 2.13e-05`` (exactly reproduced
          by a dense solve, so it is the operator and not the solver, and it
          does not move between GMRES tol 1e-9 and 1e-14).  This row gives
          6.9e-15.  Fully conjugating the row instead — the optical ``-B*, -A*``
          — breaks the RPA limit by 2.3e-03, because ``K^A = M v M^dag`` is
          Hermitian but NOT symmetric (measured ``|K^A - (K^A)^T|/|K^A| =
          1.95``).  This is the same class of defect as the historical
          ``-B, -A`` optical bug recorded below: an un-conjugated row against a
          complex-symmetric coupling.

        * ``screening=False`` (OPTICAL BSE): ``A`` is Hermitian (``A = A^H``) but
          the coupling ``B`` is complex-SYMMETRIC (``B = B^T``, NOT Hermitian).
          The physical para-Hermitian Casida operator is ``[[A, B], [-B*, -A*]]``
          (Onida-Reining-Rubio; Rohlfing-Louie), whose spectrum is REAL with
          +-omega pairs, so ``Y_out = -B* X - A* Y``.  ``B* X = conj(B conj(X))``
          and ``A* Y = conj(A conj(Y))`` reuse the SAME appliers on conjugated
          inputs (operator ingredients unchanged), then conjugate the result — no
          new kernel.  The naive ``-B X - A Y`` gives a COMPLEX (unphysical)
          spectrum for complex ``B`` and was the historical, never-value-validated
          bug (PHASE2_LOG "non-TDA eigensolvers", first checked numbers)."""
        if screening and not include_W:
            AY = _apply_A(Y, psi_c_X, psi_c_Y, psi_v_X, psi_v_Y, eps_c, eps_v,
                          W_R, V_q0, M_X, psi_cW_X, psi_vW_Y)
            BX = _apply_B(X, psi_c_X, psi_c_Y, psi_v_X, psi_v_Y, W_R, V_q0, M_X,
                          psi_cW_X, psi_cW_Y, psi_vW_X, psi_vW_Y)
            return -BX - AY
        if screening:
            # Ring parts un-conjugated (the Hartree dyad's bottom row), direct
            # parts conjugated (the ladder rung's anti-resonant blocks).  The
            # conjugated appliers reuse the SAME kernels on conjugated inputs —
            # no second W path, so nothing can drift between the two rows.
            AY = (apply_D_term(Y, eps_c, eps_v)
                  + apply_V_ring_only(Y, psi_c_Y, psi_v_Y, M_X, V_q0)
                  - jnp.conj(_w_term_A(jnp.conj(Y), psi_cW_X, psi_vW_Y, W_R)))
            BX = (apply_V_ring_only(X, psi_c_Y, psi_v_Y, M_X, V_q0)
                  - jnp.conj(_w_term_B(jnp.conj(X), psi_cW_X, psi_cW_Y,
                                       psi_vW_X, psi_vW_Y, W_R)))
            return -BX - AY
        AsY = jnp.conj(_apply_A(jnp.conj(Y), psi_c_X, psi_c_Y, psi_v_X, psi_v_Y,
                                eps_c, eps_v, W_R, V_q0, M_X, psi_cW_X, psi_vW_Y))
        BsX = jnp.conj(_apply_B(jnp.conj(X), psi_c_X, psi_c_Y, psi_v_X, psi_v_Y,
                                W_R, V_q0, M_X, psi_cW_X, psi_cW_Y,
                                psi_vW_X, psi_vW_Y))
        return -BsX - AsY

    def _impl_core_fused(X_full, psi_c_X, psi_c_Y, psi_v_X, psi_v_Y, eps_c,
                         eps_v, W_R, V_q0, M_X, psi_cW_X, psi_cW_Y, psi_vW_X,
                         psi_vW_Y):
        """The LADDER screening non-TDA matvec with TWO rung FFT chains, not four.

        ``apply_W_from_T`` is linear in ``T`` and takes the SAME
        ``(psi_cW_X, psi_vW_Y, W_R)`` in all four of the operator's rung terms,
        and each ROW's two terms enter with the same sign, so their ``T`` may be
        summed BEFORE the ``T -> T_R -> U_R -> U_q -> U`` chain instead of after:

            X_out = D X + V X + V Y      −      W[ T_A(X)      + T_B(Y)      ]
            Y_out = −(V X + D Y + V Y)   + conj W[ T_B(conj X) + T_A(conj Y) ]

        which is ``_apply_A + _apply_B`` and ``_antiresonant_row`` term for
        term — the conjugation pattern is theirs and is reproduced here, not
        re-derived (the ring parts un-conjugated because the Hartree dyad's
        bottom row is ``-K^A``; the direct parts conjugated because the ladder
        rung's anti-resonant blocks are the complex conjugates of the resonant
        ones — the reasoning is in ``_antiresonant_row``'s docstring, which
        remains the ONE place it is written down).

        WHY IT IS WORTH A SECOND SPELLING OF THE ROW.  The chain's
        ``(nb, mu, nu, s, s, k)`` buffer is the matvec: 95-96 % of the ladder
        matvec is the direct rung and ~3/4 of a rung term is this chain, so
        4 -> 2 applications is the single largest per-iteration saving on the
        path.  MEASURED (opt_integration, 2026-08-16) — see the arm's report.

        THE DUPLICATION IS GATED, NOT ASSUMED.  ``bench_w_ladder_integration
        --mode fuse`` applies both cores to the same random block and requires
        agreement at 1e-12; ``fuse_ladder_rung=False`` selects the unfused core
        for that A/B and for any bisection.  If the sign/convention forks move
        ``_antiresonant_row``, that cell goes RED rather than the two rows
        drifting silently.

        TWO -> ONE WAS TRIED AND IS REFUTED.  The obvious next step is to stack
        the two surviving applications: they are INDEPENDENT operands of the
        SAME operator (same ``psi``, same ``W_R``, different ``T``), XLA
        schedules them strictly serially under every flag, and
        ``apply_W_from_T`` is row-count-agnostic on both arms — so
        ``jnp.concatenate([T_res, T_anti], axis=0)``, ONE application, and two
        slices back is a legal and bit-identical rewrite.  MEASURED on the
        gnppm fixture (n_rmu 399, 3x3x1, nspinor 2; one A100, BFC; in-process
        A/B, and the build order swapped as a control):

            arm                     nb=1              nb=4
            XLA chain           +1.7 % faster     +0.6 % faster
            FUSED k-minor       -15 %  SLOWER     -11 %  SLOWER

        Both arms agree BIT-EXACTLY with the unstacked path (rel 0.0e+00), so
        this is a scheduling result and not a numerics one.  The sign flips
        because of what consumes ``T``: the XLA chain's first op is a
        reshape/FFT that XLA fuses the ``concatenate`` INTO (peak unchanged),
        while the fused kernel is a CUSTOM CALL, whose operand must be
        materialized — the concatenate becomes a real copy of the doubled
        tile, measured as +152 MiB peak at nb=1 (788.0 vs 636.1) and +0.44
        ms/col, i.e. about two HBM passes over the pair.  Since ``auto`` (the
        kernel) is the default and the faster arm, a win on the slower arm
        does not pay for a 15 % loss on the default one, and the stacking is
        NOT taken.  Evidence:
        ``reports/screening_diagrams_wbse/evidence/rung_pair_batch/``.
        The general lesson, which outlives this rung: batching operands across
        a fused custom call has to pay for materializing the batched operand,
        and only an arm whose first consumer is fusable gets that for free.
        """
        X, Y = X_full[0], X_full[1]
        DX = apply_D_term(X, eps_c, eps_v)
        DY = apply_D_term(Y, eps_c, eps_v)
        VX = apply_V_ring_only(X, psi_c_Y, psi_v_Y, M_X, V_q0)
        VY = apply_V_ring_only(Y, psi_c_Y, psi_v_Y, M_X, V_q0)
        T_res = (_T_term_A(X, psi_cW_X, psi_vW_Y)
                 + _T_term_B(Y, psi_cW_Y, psi_vW_X))
        T_anti = (_T_term_B(jnp.conj(X), psi_cW_Y, psi_vW_X)
                  + _T_term_A(jnp.conj(Y), psi_cW_X, psi_vW_Y))
        W_res = apply_W_from_T(T_res, psi_cW_X, psi_vW_Y, W_R)
        W_anti = jnp.conj(apply_W_from_T(T_anti, psi_cW_X, psi_vW_Y, W_R))
        X_out = DX + VX + VY - W_res
        Y_out = -VX - DY - VY + W_anti
        return jnp.stack([X_out, Y_out], axis=0)

    def _impl_core(X_full, psi_c_X, psi_c_Y, psi_v_X, psi_v_Y, eps_c, eps_v,
                   W_R, V_q0, M_X, psi_cW_X, psi_cW_Y, psi_vW_X, psi_vW_Y):
        if fuse_ladder_rung and screening and include_W:
            return _impl_core_fused(X_full, psi_c_X, psi_c_Y, psi_v_X, psi_v_Y,
                                    eps_c, eps_v, W_R, V_q0, M_X,
                                    psi_cW_X, psi_cW_Y, psi_vW_X, psi_vW_Y)
        X = X_full[0]
        Y = X_full[1]
        AX = _apply_A(X, psi_c_X, psi_c_Y, psi_v_X, psi_v_Y, eps_c, eps_v,
                      W_R, V_q0, M_X, psi_cW_X, psi_vW_Y)
        BY = _apply_B(Y, psi_c_X, psi_c_Y, psi_v_X, psi_v_Y, W_R, V_q0, M_X,
                      psi_cW_X, psi_cW_Y, psi_vW_X, psi_vW_Y)
        X_out = AX + BY
        Y_out = _antiresonant_row(X, Y, psi_c_X, psi_c_Y, psi_v_X, psi_v_Y,
                                  eps_c, eps_v, W_R, V_q0, M_X,
                                  psi_cW_X, psi_cW_Y, psi_vW_X, psi_vW_Y)
        return jnp.stack([X_out, Y_out], axis=0)

    if ladder_rung_slots:
        # LADDER on a build_finite_q_data payload: the rung's psi are four
        # EXTRA runtime operands (rolled, UN-flipped) — see
        # bse_feast.ladder_matvec_operands and _w_term_A's operand contract.
        def _matvec_impl(X_full, psi_c_X, psi_c_Y, psi_v_X, psi_v_Y, eps_c,
                         eps_v, W_R, V_q0, M_X, M_Y,
                         psi_cW_X, psi_cW_Y, psi_vW_X, psi_vW_Y):
            # M_X: hoisted decode-side exchange pair amplitude (audit P3).
            # M_Y is unused here — kept for a uniform matvec signature.
            return _impl_core(X_full, psi_c_X, psi_c_Y, psi_v_X, psi_v_Y,
                              eps_c, eps_v, W_R, V_q0, M_X,
                              psi_cW_X, psi_cW_Y, psi_vW_X, psi_vW_Y)

        matvec = jax.jit(
            _matvec_impl,
            in_shardings=(
                sh.X_full,
                sh.psi_x, sh.psi_y, sh.psi_x, sh.psi_y,
                sh.eps, sh.eps, sh.W, sh.V,
                sh.psi_x, sh.psi_y,
                sh.psi_x, sh.psi_y, sh.psi_x, sh.psi_y,
            ),
            out_shardings=sh.X_full,
        )
    else:
        def _matvec_impl(X_full, psi_c_X, psi_c_Y, psi_v_X, psi_v_Y, eps_c,
                         eps_v, W_R, V_q0, M_X, M_Y):
            # M_X: hoisted decode-side exchange pair amplitude (audit P3), shared by
            # the A and B blocks. M_Y is unused here — kept for a uniform matvec
            # signature.  The rung (if any) consumes the density psi arrays —
            # correct for every raw payload (the only kind these operators see).
            return _impl_core(X_full, psi_c_X, psi_c_Y, psi_v_X, psi_v_Y,
                              eps_c, eps_v, W_R, V_q0, M_X,
                              psi_c_X, psi_c_Y, psi_v_X, psi_v_Y)

        matvec = jax.jit(
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
                sh.psi_x,
                sh.psi_y,
            ),
            out_shardings=sh.X_full,
        )
    if not return_half_appliers:
        return matvec

    # The two HALF-operator appliers, EXPOSED rather than duplicated.  Until now
    # they were reachable only through ``_matvec_impl``, which evaluates FOUR of
    # them per call -- ``A X``, ``B Y``, ``conj(A conj(Y))``, ``conj(B conj(X))``
    # -- so a caller that wants a single block (the dense build; any matrix-free
    # route) paid for two applications against a ZERO block.  Same closures,
    # same operator ingredients, same shardings as the corresponding terms
    # inside the full matvec: nothing is re-derived and there is no second copy
    # of the kernel to drift.
    # Half-appliers serve raw payloads only (the ladder_rung_slots combination
    # refuses above), so the rung's psi operands ARE the density ones here.
    def _apply_A_raw(X, psi_c_X, psi_c_Y, psi_v_X, psi_v_Y, eps_c, eps_v,
                     W_R, V_q0, M_X):
        return _apply_A(X, psi_c_X, psi_c_Y, psi_v_X, psi_v_Y, eps_c, eps_v,
                        W_R, V_q0, M_X, psi_c_X, psi_v_Y)

    def _apply_B_raw(X, psi_c_X, psi_c_Y, psi_v_X, psi_v_Y, W_R, V_q0, M_X):
        return _apply_B(X, psi_c_X, psi_c_Y, psi_v_X, psi_v_Y, W_R, V_q0, M_X,
                        psi_c_X, psi_c_Y, psi_v_X, psi_v_Y)

    apply_A = jax.jit(
        _apply_A_raw,
        in_shardings=(sh.X, sh.psi_x, sh.psi_y, sh.psi_x, sh.psi_y,
                      sh.eps, sh.eps, sh.W, sh.V, sh.psi_x),
        out_shardings=sh.X,
    )
    apply_B = jax.jit(
        _apply_B_raw,
        in_shardings=(sh.X, sh.psi_x, sh.psi_y, sh.psi_x, sh.psi_y,
                      sh.W, sh.V, sh.psi_x),
        out_shardings=sh.X,
    )
    return matvec, apply_A, apply_B
