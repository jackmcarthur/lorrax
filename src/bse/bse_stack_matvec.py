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
``n_trials`` (the memory hog).  Here the W-term is ONE ``shard_map`` whose body
is a ``lax.scan`` over the trial axis, so XLA reuses the body's scratch across
iterations: exactly ONE ``T``-family is alive regardless of ``n_trials``.  A
Python-unrolled or ``fori_loop``-over-trials-inside-``jit`` would pile up
``n_trials`` live ``T`` slots (the known slot-pile-up failure mode,
``feedback_path_d_scaffolding_pattern``); the scan avoids it.  Collectives run
per trial inside the scan body — the memory-for-comm trade the design chose.

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
    OUT OF SCOPE (B2 malformed B-encode is unfixed).  When B2 lands it should
    reuse this module's ``_w_stack`` encode/decode rather than its own.
"""
from __future__ import annotations

import os

import jax
import jax.numpy as jnp
import numpy as np
from jax import lax
from jax.sharding import Mesh, PartitionSpec as P

from common.shard_map import shard_map as _shard_map_fn

from common.fft_helpers import local_fftn3, local_ifftn3
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
#   yhoist  Lift the two 'y'-axis collectives OUT of the per-trial scan.
#           ``all_gather(X_b,'y')`` and the final ``psum_scatter(...,'y')``
#           are the two SMALL collectives (the operand is (c_loc, v, nk) --
#           2 KB per trial at P=64, vs 426 KB for the 'x' pair), so batching
#           them over the trial axis costs 16 KB per rank in total and
#           removes 2 of the 4 collectives per trial.  The 'x' pair is
#           deliberately NOT hoisted: batching those needs an
#           (n_trials, c, nk, ns, nu_loc) staging buffer -- 3.4 MB per rank,
#           an n_trials-fold replication of a T-adjacent intermediate, which
#           is exactly the memory-for-comm trade the owner has vetoed.  The
#           accounting is per-rank bytes, and it is the whole argument for
#           why one half of this is allowed and the other is not.
#           MEASURED (job 7883166): ALONE it is 1.007x on a full production Q
#           -- inside the 3.42% run-to-run spread the same job measured by
#           running its baseline cell twice, i.e. NOTHING.  Combined with
#           the removed dense-k lever it measured 2.319x, because once the
#           FFT is gone the collectives are a much larger share of what is
#           left.  Report it that way: it is not a lever on its own.
#
#   krep    REPLICATE the Krylov vectors.  This one is NOT in this module --
#           it is honoured by the two eigensolver bridges
#           (``exciton_bands.build_path_solver`` and
#           ``bse_lanczos.solve_bse_sharded``), which are the only places that
#           know both the mesh and the flat (block, n_flat) Krylov layout.  It
#           lives in THIS enum so there is one dial and one grammar, not three.
#           The pair-space dimension n_flat = nc_pad*nv_pad*nk is 1024 on the
#           MoS2 4x4 deck -- four orders below N_mu -- yet the flat Krylov axis
#           carries no sharding constraint, so GSPMD tiles it across all 64
#           devices and every reorthogonalisation dot product becomes an
#           all-reduce.  At max_iter=40 with full reorthogonalisation that is
#           sum_j (j+1) = 820 all-reduces of a 1 KB tile per Q, plus a gather
#           for each QR.  Pinning the Krylov basis to P() makes all of that
#           local and leaves ONE all-gather per matvec.
#           THE COST IS NOT SCALE-FREE and must be stated every time this is
#           recommended: the replicated basis is (max_iter+1)*n_flat*block*16
#           bytes on EVERY rank -- 5.4 MB at n_flat=1024, but 525 MB at
#           n_flat=1e5.  It is right for a small pair space and wrong for a
#           large one, so it is a dial and not a default.
#
# None of the three changes the number of live (nk, mu_loc, nu_loc, ns^2)-class
# intermediates, which stays at the one ``T_b`` family documented below.
_MATVEC_OPTS = ("yhoist", "krep")   # densek REMOVED 2026-07-31, owner directive


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
    use_yhoist = "yhoist" in opts

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

        def _body(carry, X_b):                       # X_b: (c_loc, v_loc, nk)
            # encode: T_b[μ,ν,t,s,k] = Σ_c ψ_c[k,c,t,μ] Σ_v conj(ψ_v[k,v,s,ν]) X_b
            # With ``yhoist`` the 'y' all-gather already happened OUTSIDE the
            # scan, so X_b arrives as (c_loc, v_full, nk) — the same operand
            # ``Xv`` is below, one collective per BLOCK instead of per trial.
            Xv = (X_b if use_yhoist
                  else lax.all_gather(X_b, "y", axis=1, tiled=True))  # (c_loc, v_full, nk)
            R = jnp.einsum("kvsN,cvk->cksN", jnp.conj(psi_v_Y), Xv)  # (c_loc, nk, ns, ν_loc)
            Rc = lax.all_gather(R, "x", axis=0, tiled=True)          # (c_full, nk, ns, ν_loc)
            T_b = jnp.einsum("kctM,cksN->MNtsk", psi_c_X, Rc)        # (μ_loc, ν_loc, ns, ns, nk)
            mu_loc, nu_loc, ns = T_b.shape[0], T_b.shape[1], T_b.shape[2]

            # conv: U_b = (1/√Nk) Σ_q W_q T_b[..., k−q]  (ortho ifft_k · W_R · fft_k)
            #
            # THIS IS AN FFT AND IT STAYS AN FFT.  A dense (nk x nk) DFT
            # contraction gives the same numbers and measured 2.3x faster on
            # this deck, and it was REMOVED on 2026-07-31 by owner directive:
            # the dense form is O(nk^2) where the FFT is O(nk log nk), so it
            # is a win only because nk = 16 here and it inverts at the
            # thousand-k-point sizes LORRAX is being built for.  Optimising
            # against the current deck's nk is the error family this project
            # calls deck-tuning.  Do not reintroduce it, under any name, on
            # any measurement.  If the k-transform is a bottleneck the answer
            # is a better FFT (batching, an FFI handler, fewer dispatches),
            # never a denser algorithm.
            T_k = T_b.reshape(mu_loc, nu_loc, ns, ns, nkx, nky, nkz)
            T_R = local_ifftn3(T_k, axes=(4, 5, 6), norm="ortho")
            U_R = W_R[:, :, None, None, :, :, :] * T_R
            U_b = local_fftn3(U_R, axes=(4, 5, 6), norm="ortho").reshape(
                mu_loc, nu_loc, ns, ns, nk)

            # decode: (WX)_b = (1/√Nk) Σ_{μ,ν,t,s} conj(ψ_c) ψ_v U_b.  psum_scatter
            # completes the μ-sum while scattering c→x, then the ν-sum while
            # scattering v→y — no replicated (c_full, v_full) buffer survives.
            A = lax.psum_scatter(
                jnp.einsum("kctM,MNtsk->cNsk", jnp.conj(psi_c_X), U_b),
                "x", scatter_dimension=0, tiled=True)               # (c_loc, ν_loc, ns, nk)
            WXcv = jnp.einsum("kvsN,cNsk->cvk", psi_v_Y, A)         # (c_loc, v_full, nk)
            if not use_yhoist:
                WXcv = lax.psum_scatter(
                    WXcv, "y", scatter_dimension=1, tiled=True)     # (c_loc, v_loc, nk)
            return carry, WXcv / sqrt_nk

        if use_yhoist:
            # ONE 'y' all-gather for the whole block instead of n_trials of
            # them.  Operand (n_trials, c_loc, v_loc, nk) -> (…, v_full, …):
            # 16 KB per rank at P=64, against the 11.08 MB T_b the scan body
            # already holds.  The scan carries no extra T-class buffer.
            X = lax.all_gather(X, "y", axis=2, tiled=True)
        _, WX = lax.scan(_body, None, X)             # WX: (n_trials, c_loc, v_*, nk)
        if use_yhoist:
            WX = lax.psum_scatter(WX, "y", scatter_dimension=2, tiled=True)
        return WX

    w_stack = _shard_map_fn(
        _w_stack,
        mesh=mesh_xy,
        in_specs=(P(None, "x", "y", None), P(None, None, None, "x"),
                  P(None, None, None, "y"), P("x", "y", None, None, None)),
        out_specs=P(None, "x", "y", None),
    )

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
