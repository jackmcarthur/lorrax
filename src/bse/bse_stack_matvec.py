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

The exchange (V) term is NOT stack-specific: it is the shared
``bse_ring_comm.bse_exchange_gspmd`` (2 all-reduce), reused verbatim by the ring
TDA / non-TDA matvecs (audit P2/C1) — single source of truth. Only the W-term
encode/decode differs stack-vs-ring.

Dispatch (audit P-NT): ``bse_lanczos.solve_bse_sharded`` routes narrow solves
(block width ≤2: single-vector / narrow-block Lanczos) to the ring and wide ones
(≥3: block-Lanczos, Davidson subspaces) here — the measured crossover is nt≈2-3
(the ring batches trials → collectives fixed in width, ~1.5× faster + never larger
at nt1; the stack scans → memory flat, wins at nt≥3).

Retirement plan (NOTED, not yet executed) — the gather/simple TDA matvecs existed
only to bound ``T``'s peak; this scan bounds it strictly better, so they are
superseded.  Still live pending a follow-up:
  * ``bse_ring_comm.build_bse_ring_matvec`` (ring TDA) — the narrow-nt dispatch
    target above, ``bse_feast.estimate_spectral_bounds_sharded`` (spectral-bound
    Lanczos), and the equality gates.  KEPT (P-NT: optimal at nt≤2).
  * ``bse_ring_comm.build_bse_ring_matvec_full`` (non-TDA S=[[A,B],[-B†,-A†]]):
    the W(0)/finite-q resolvent operator.  Its W-term B-encode still uses the ring
    ``encode_T_B``; when that is retired it should reuse this module's ``_w_stack``
    encode/decode.  The V-term is already unified (``bse_exchange_gspmd``).
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
from jax import lax
from jax.sharding import Mesh, PartitionSpec as P

try:
    from jax import shard_map as _shard_map_fn
except ImportError:  # pragma: no cover - older JAX
    from jax.experimental import shard_map as _shard_map_mod
    _shard_map_fn = _shard_map_mod.shard_map

from common.fft_helpers import local_fftn3, local_ifftn3
from .bse_ring_comm import bse_exchange_gspmd, make_bse_shardings


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
            Xv = lax.all_gather(X_b, "y", axis=1, tiled=True)        # (c_loc, v_full, nk)
            R = jnp.einsum("kvsN,cvk->cksN", jnp.conj(psi_v_Y), Xv)  # (c_loc, nk, ns, ν_loc)
            Rc = lax.all_gather(R, "x", axis=0, tiled=True)          # (c_full, nk, ns, ν_loc)
            T_b = jnp.einsum("kctM,cksN->MNtsk", psi_c_X, Rc)        # (μ_loc, ν_loc, ns, ns, nk)
            mu_loc, nu_loc, ns = T_b.shape[0], T_b.shape[1], T_b.shape[2]

            # conv: U_b = (1/√Nk) Σ_q W_q T_b[..., k−q]  (ortho ifft_k · W_R · fft_k)
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
            WXcv = lax.psum_scatter(
                jnp.einsum("kvsN,cNsk->cvk", psi_v_Y, A),
                "y", scatter_dimension=1, tiled=True)               # (c_loc, v_loc, nk)
            return carry, WXcv / sqrt_nk

        _, WX = lax.scan(_body, None, X)             # WX: (n_trials, c_loc, v_loc, nk)
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
        # ── D term: (ε_c − ε_v) · X  (batched, local) ──────────────────────────
        delta_E = eps_c.T[None, :, None, :] - eps_v.T[None, None, :, :]
        D_term = lax.with_sharding_constraint(delta_E * X, sh.X)

        # ── V term: B1 dense exchange via the shared GSPMD form (2 all-reduce). ──
        # Single source of truth with the ring matvecs (bse_ring_comm).
        VX = bse_exchange_gspmd(X, M_Y, M_X, V_q0, sh, nk)

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
