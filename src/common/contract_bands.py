"""contract_bands_block_reshard — THE multi-stage band projection + reshard.

Single source of truth (owner directive, 2026-07-28) for the pattern

    out[extra?, k, m, n] = Σ_{s,μ} Σ_{s',ν}  conj(ψ_left)[k, m, s, μ]
                           · O[extra?, k, s, μ, s', ν] · ψ_right[k, s', ν, n]

executed as a TWO-STAGE psum_scatter chain on the 2-D ('x','y') mesh, so
that **no (m, n, k)-, (m, μ)- or (μ, μ)-sized object is ever materialized
on one rank** — that is this primitive's raison d'être (LORRAX scaling
target: thousands of low-memory processes; an N_μ² or band-cube tile on a
single rank is the failure mode this module exists to make structurally
impossible).  The output leaves the chain already block-sharded
(m on 'x', n on 'y'); every partial bigger than the final tile lives only
as a rank-local shard between the two collectives.

Structure (per rank, inside one shard_map)::

    right    = einsum(O_local, ψ_right_local)     contract (s', ν_Y-local)
    right_rs = psum_scatter(right, 'y', dim=n)    LARGE payload, 'y' groups
    left     = einsum(conj(ψ_left_local), right_rs)  contract (s, μ_X-local)
    out      = psum_scatter(left, 'x', dim=m)     small payload, 'x' groups

Adopters (2026-07-28): the two Σ_c(τ) projection tails in
``gw.ppm_tau_kernel`` (two-channel crossing plan + merged Laplace plan).
Slated (owner review pending, wk_REL/contract_bands_notes.md §BSE):
``bse.bse_stack_matvec._w_stack`` decode, the ``bse_ring_comm`` matvec
family, ``vq_interp`` V_Q assembly.

Encoded policies (each measured; evidence cited inline)
-------------------------------------------------------
1. **Large-payload-on-the-node-local-axis** (owner-approved axis-order
   swap, 2026-07-28): the ν/'y'-side contraction runs FIRST, so the LARGE
   partial — the m-full ``(k, s, μ_X_loc, n)`` block — reduce-scatters
   over the 'y' mesh axis, whose replica groups are CONSECUTIVE ranks
   (node-local pairs at 2 ranks/node on the production layout; HLO
   module_0912 groups {8x..8x+7} vs stride-8 {y, y+8, ...} for 'x').
   Only the small final ``(k, m, n/p_y)`` block rides the zero-locality
   'x' groups.  This is why ψ_left is the 'x'/m-side operand and ψ_right
   the 'y'/n-side operand — the primitive REFUSES a mesh whose minor
   (consecutive-rank) axis is not the 'y' axis it scatters the large
   payload over.
2. **Stacked collectives** (AK.9): every channel/extra slice rides ONE
   collective per mesh axis (stack-then-scatter), never one collective
   per slice.  Bit-exact by construction (elementwise rank-sum is
   indifferent to concatenation); message count is flat in the stack
   extent.
3. **De-promoted f64-split lowering where operands are real**
   (wk_REL/RESHARD_OVERHEAD_MEMO.md Sec. 4.4, HLO-proven; measured
   project_rs 43.2 → 38.7 s at nb=128/P=64, job 7878942): whenever the
   large right contraction would be mixed f64 × c128, XLA:CPU (and GPU)
   PROMOTES the real operand to c128 — a ~400 MB materialization per
   channel at production shape — and runs a full complex GEMM at 2× the
   required flops.  This module instead splits the COMPLEX ψ_right into
   its f64 parts and issues pure-f64 dgemms + one ``lax.complex``
   recombine.  Applied automatically when ``O`` (or a derived channel)
   is real; genuinely complex × complex contractions are left as single
   complex GEMMs — the f64 split was tried there and REFUTED by
   measurement (Eigen dgemm ~172 GF/s is per-flop BELOW its zgemm's
   295 at these shapes; job 7878942, refutation recorded in
   ``gw.ppm_tau_kernel._project_x_local``'s history).
4. **The impl=mpi mesh-clique warm-up** (corrected 2026-07-29).  Under
   ``JAX_CPU_COLLECTIVES_IMPLEMENTATION=mpi`` jaxlib refuses to CREATE a
   communicator from any thread but the MPI-initialising one
   (``MPI_Is_thread_main`` in
   ``xla::cpu::MpiCollectives::CreateCommunicators``), and XLA:CPU's
   parallel ``ThunkExecutor`` issues collective thunks from intra-op pool
   workers — so a clique whose FIRST use is inside a real jitted program
   dies on every rank.  The factory therefore calls
   :func:`common.collectives.warm_mesh_cliques` before returning, which
   creates each clique once from the main thread; XLA then serves every
   later acquisition from its process-global clique cache.

   This SUPERSEDES the earlier ``ensure_grouped_collectives_ready``, which
   warmed the WORLD clique only.  The controls (job 7881053) show world-only
   FAILS, as do x-only and x+y-without-world; only x+y+world passes.  The
   old "world-collective-first contract" was right that warm-up matters and
   wrong about which device sets to warm — the caching is **per-clique**.
5. **Divisibility guard with an actionable refusal**: the two
   psum_scatters split m over p_x and n over p_y; an indivisible window
   would crash cryptically deep inside psum_scatter.  The wrapper
   converts that into an error that names the fix (pad m and n
   INDEPENDENTLY per axis — never to the p_x·p_y product, which wastes
   up to 3.16× tile; audit fix/zq 2026-07-28).

Gated CPU GEMM body (LORRAX_BANDS_GEMM_FFI; AUTO default 2026-07-29)
--------------------------------------------------------------------
The large right contraction is the measured wall of this pattern — 71% of
the Σ project_rs row, running 4.3–7.3× below the node's BLAS roofline
through XLA:CPU's Eigen dots (295 GF/s promoted zgemm / ~172 GF/s split
dgemm vs 1263 GF/s MKL, memo Sec. 4.4/4.5; the bare Eigen dot saturates
1.6–1.9× below vendor BLAS at full threads — jobs 7879008/7879010).
The dial routes ONLY that right contraction through the vendor-BLAS GEMM
host FFI handler; everything else — channel algebra, collectives, the
small left dots (1.6e-3 of the right's flops, measured) — is untouched.

The dial ITSELF is a microservice: ``ffi.mklblas`` owns its grammar,
platform resolution, capability probe, announcements, refusals and the
``ffi_call`` (handler ``lorrax_mklblas_gemm_batch``,
src/ffi/cpp/mklblas).  ``docs/dev/vendor_gemm_service.md`` is its
contract, ``docs/dev/ffi_gate_contract.md`` the gate doctrine it
implements.  Default is AUTO (owner order 2026-07-29, doctrine #8:
capability detection, not policy) — ON when the platform is CPU AND the
handler resolves in the host .so, announced once, and quietly native
everywhere it cannot apply.

What belongs to THIS module, and is enforced here, is only what a GEMM
service cannot know: WHICH contraction is routed, and that
``extra="minor"`` is excluded — a fact about this primitive's operand
layout, not about BLAS (the contracted axis is not reachable by a strided
batched GEMM without a full-tile transpose copy, which would cost what
the handler saves).  Under AUTO the minor order quietly keeps the XLA
plan; under an explicit ``=1`` it REFUSES, like every other unhonorable
explicit request.  ``input_output_aliases``: deliberately NONE — a GEMM
output (B, M, N) never matches an operand buffer shape, so no alias is
legal (contrast the in-place FFT handlers).  Read at FACTORY time:
kernel caches must key on :func:`bands_gemm_ffi_enabled` (ppm_tau_kernel
does).

Canonical operand layout (global shapes → mesh specs)
-----------------------------------------------------
::

    ψ_left   (nk, m, s, μ)          P(None, None, None, 'x')
    O        (nk, s, μ, s', ν)      P(None, None, 'x', None, 'y')
      extra="leading": (E, nk, s, μ, s', ν)   P(None, None, None, 'x', None, 'y')
      extra="minor":   (nk, s, μ, s', ν, E)   P(None, None, 'x', None, 'y', None)
    ψ_right  (nk, s', ν, n)         P(None, None, 'y', None)
    out      (nk, m, n)             P(None, 'x', 'y')   (+E leading/minor)

``s``/``s'`` are the spinor axes (size 1 is fine); ``conj`` is applied to
ψ_left inside the body (ψ† σ ψ semantics; pass a pre-conjugated array if
you need the unconjugated form).  The optional ``extra`` axis is the
caller's stack — Σ channels, BSE trial blocks, a τ/ω batch — and rides
the stacked collectives of policy 2.  ``extra="leading"`` vs ``"minor"``
is a measured choice, not a style one: the unit gate microbenches both
orders at production local shapes and the winner is recorded in
wk_REL/contract_bands_notes.md (owner question — numbers, not opinions).

Envelope statement (design-envelope rule): all policies are flat in
n_atoms / N_μ / nk / nb / P; per-rank payloads scale as the SHARD sizes
(right partial ~ nb·μ/p_x per channel), never as a global tile; the
primitive lowers on CPU and GPU (shard_map + psum_scatter are
backend-neutral; the FFI dial is CPU-only and refused elsewhere).
"""

from __future__ import annotations

from typing import Callable

import jax
import jax.numpy as jnp
from jax.sharding import Mesh, PartitionSpec as P

from common.collectives import warm_mesh_cliques

__all__ = [
    "contract_bands_block_reshard",
    "bands_gemm_ffi_enabled",
    "bands_gemm_ffi_mode",
]


# ---------------------------------------------------------------------------
# Gated vendor-BLAS GEMM FFI body — the SERVICE lives in ``ffi.mklblas``
# ---------------------------------------------------------------------------
# The dial's grammar, platform resolution, capability probe, announcements
# and refusals are the microservice's (``src/ffi/mklblas/gemm.py``, on the
# shared ``ffi.common.gate.Gate``); its ``ffi_call`` is there too, per
# ``src/ffi/AGENTS.md:149-150``.  This module keeps only the two things that
# are genuinely ITS policy and cannot live in a GEMM service: WHICH
# contraction is routed (the large right one, never the left dots), and the
# ``extra="minor"`` structural exclusion, which is a fact about this
# primitive's operand layout rather than about BLAS.  Contract:
# ``docs/dev/vendor_gemm_service.md``; gate doctrine:
# ``docs/dev/ffi_gate_contract.md``.
from ffi.mklblas import (                                    # noqa: E402
    GATE as _BANDS_GEMM_GATE,
    gemm_batch as _gemm_batch_ffi,
    require_bands_gemm_ffi as _require_bands_gemm_ffi,
)


def bands_gemm_ffi_mode() -> str:
    """The LORRAX_BANDS_GEMM_FFI grammar: ``"on"`` | ``"off"`` | ``"auto"``
    (delegates to :data:`ffi.mklblas.GATE`)."""
    return _BANDS_GEMM_GATE.mode()


def bands_gemm_ffi_enabled() -> bool:
    """True when the primitive's LARGE right contraction routes through the
    vendor-BLAS GEMM host FFI handler.  DEFAULT is AUTO (2026-07-29 owner
    order): ON when the platform is CPU and the handler resolves in the
    host .so (announced once), OFF otherwise.  Read at FACTORY time —
    kernel caches must key on this (ppm_tau_kernel's pipeline/cache keys
    do), which is why the auto resolution is lexical and never initializes
    the JAX backend (gate contract, tier 1)."""
    return _BANDS_GEMM_GATE.enabled()


# ---------------------------------------------------------------------------
# The primitive
# ---------------------------------------------------------------------------

_DIVISIBILITY_MSG = (
    "contract_bands_block_reshard needs the band window divisible by the "
    "mesh: m={m} must be a multiple of p_{ax_x}={px} and n={n} of "
    "p_{ax_y}={py} (the two psum_scatters tile m over '{ax_x}' and n over "
    "'{ax_y}').  Pad m and n INDEPENDENTLY (m -> p_{ax_x}, n -> p_{ax_y}) "
    "at the caller and strip the zero pad downstream — do NOT round to a "
    "multiple of the p_{ax_x}*p_{ax_y} product (up to 3.16x tile waste; "
    "audit fix/zq 2026-07-28).  Sigma callers: gw.ppm_sigma.pad_sigma_window "
    "is the existing helper.{hint}")


def contract_bands_block_reshard(
    mesh_xy: Mesh,
    *,
    channels: str = "none",
    extra: str = "none",
    axes: tuple[str, str] = ("x", "y"),
    divisibility_hint: str = "",
) -> Callable:
    """Build the band projection + reshard primitive (module docstring).

    Parameters
    ----------
    mesh_xy
        The 2-D device mesh.  Its MINOR (last-named, consecutive-rank)
        axis must be ``axes[1]`` — policy 1 scatters the large partial
        over that axis and refuses an inverted mesh rather than silently
        shipping the big payload over the zero-locality groups.
    channels
        ``"none"``: one contraction chain at O's own dtype (complex O →
        single complex chain, exactly the merged Laplace plan; REAL O →
        de-promoted f64-split chain, policy 3).
        ``"split_reim"``: O must be complex; it is split elementwise into
        (Re O, Im O) BEFORE projection and each real channel rides its
        own de-promoted chain — the two-channel (S_R, S_I) plan required
        by consumers that weight the channels independently (crossing
        windows; channel algebra: gw.ppm_tau_kernel + manual §7.5).
        Returns the tuple ``(S_R, S_I)``, both complex.  Incompatible
        with ``extra`` (refused): stack the channel pair yourself as a
        real leading-extra operand if you need both.
    extra
        ``"none"`` | ``"leading"`` | ``"minor"`` — position of the
        caller's stack axis E (see canonical layout).  Both orders are
        first-class so the choice stays a measurement
        (wk_REL/contract_bands_notes.md records the microbench).
        ``"minor"`` is refused under the FFI GEMM dial: the contracted
        axis is then not GEMM-reachable without a full-tile transpose
        copy — use ``"leading"`` there.
    axes
        Mesh axis names ``(ax_x, ax_y)``; ax_x shards μ/m, ax_y shards
        ν/n.  Default matches every production mesh.
    divisibility_hint
        Extra caller-specific text appended to the divisibility refusal.

    Returns
    -------
    ``project(psi_left, O, psi_right)`` — a shard_map'd callable, safe to
    jit / trace into larger kernels.  Output shardings per the canonical
    layout table.
    """
    from jax.experimental.shard_map import shard_map

    if channels not in ("none", "split_reim"):
        raise ValueError(
            f"channels must be 'none' or 'split_reim', got {channels!r}")
    if extra not in ("none", "leading", "minor"):
        raise ValueError(
            f"extra must be 'none', 'leading' or 'minor', got {extra!r}")
    if channels == "split_reim" and extra != "none":
        raise ValueError(
            "channels='split_reim' with an extra stack axis is not "
            "supported: the re/im channel split IS the stacked axis of "
            "that plan.  Stack (Re O, Im O) yourself as a real "
            "extra='leading' operand of channels='none' if you need an "
            "additional batch dimension.")
    ax_x, ax_y = axes
    names = tuple(mesh_xy.axis_names)
    if ax_x not in names or ax_y not in names:
        raise ValueError(
            f"mesh axes {names} do not contain axes={axes!r}")
    if names[-1] != ax_y:
        raise ValueError(
            f"contract_bands_block_reshard: the mesh's minor axis is "
            f"{names[-1]!r} but the large partial must reduce-scatter over "
            f"{ax_y!r} — on the standard process-ordered device layout only "
            f"the LAST mesh axis has consecutive-rank (node-local) replica "
            f"groups, and shipping the large payload over strided groups is "
            f"the exact inversion this primitive exists to prevent (memo "
            f"Sec. 6.1 finding #1).  Build the mesh with {ax_y!r} minor, or "
            f"pass axes=(major, minor) matching your mesh.")

    ffi_mode = bands_gemm_ffi_mode()
    use_ffi = bands_gemm_ffi_enabled()
    if use_ffi and ffi_mode == "on":
        # Explicit request: announce-or-REFUSE, never silently downgraded.
        _require_bands_gemm_ffi(mesh_xy)
        if extra == "minor":
            raise RuntimeError(
                "LORRAX_BANDS_GEMM_FFI does not support extra='minor': with "
                "the stack axis minor the contracted (s', ν) axis is not "
                "reachable by a strided batched GEMM without a full-tile "
                "transpose copy (which would cost what the handler saves).  "
                "Use extra='leading' with the FFI dial, or unset "
                "LORRAX_BANDS_GEMM_FFI for the minor order (explicit "
                "requests are never silently downgraded).")
    elif use_ffi:
        # AUTO (the default): capability detection, not policy — quietly
        # keep the native lowering where the dial cannot apply.  The gate
        # owns the mesh-platform half (non-CPU mesh: OFF, silent BY DESIGN
        # and with the reason recorded in the gate's
        # ``silent_platform_demote`` field — XLA:GPU's dot lowering already
        # hits cuBLAS, which is optimal); THIS module owns extra='minor',
        # which is a fact about the primitive's operand layout, not about
        # BLAS: the contracted axis is not GEMM-reachable there (see the
        # explicit-mode refusal above), so the minor order keeps the XLA
        # plan.  The probe + AUTO-ON announce already ran inside
        # bands_gemm_ffi_enabled().
        if _BANDS_GEMM_GATE.resolve(mesh_xy) is None or extra == "minor":
            use_ffi = False

    p_x = mesh_xy.shape[ax_x]
    p_y = mesh_xy.shape[ax_y]

    # -- rank-local GEMM helpers ---------------------------------------
    # The LARGE right contraction, optionally through the FFI handler.
    # Reshapes around the FFI call are index-preserving (contiguous
    # row-major flattening of adjacent axes) — free at the XLA level.

    def _ffi_dtypes_ok(o_l, psi_l):
        """The GEMM handler serves ALL FOUR BLAS precisions since the
        2026-07-29 owner order — f64/f32/c128/c64, dispatched on the
        buffer dtype onto cblas_{d,s,z,c}gemm[_batch].  The BSE fp32-GMRES
        (complex64) path therefore RIDES the handler now; it used to be a
        capability boundary (auto fell back, explicit ``=1`` refused).

        What remains excluded is only a genuinely unserveable dtype (f16,
        bf16, c256, or a mismatched pair that the de-promotion policy
        should have split upstream).  Explicit mode still REFUSES those
        with the fix named; AUTO still quietly keeps the XLA lowering."""
        ok = (o_l.dtype == psi_l.dtype
              and o_l.dtype in (jnp.float64, jnp.float32,
                                jnp.complex128, jnp.complex64))
        if ok or ffi_mode != "on":
            return ok
        raise TypeError(
            f"LORRAX_BANDS_GEMM_FFI: the vendor-BLAS GEMM host handler "
            f"serves f64/f32/c128/c64, got O {o_l.dtype} × ψ "
            f"{psi_l.dtype}.  A MISMATCHED pair means the de-promotion "
            f"policy did not split a mixed real/complex contraction "
            f"upstream (a bug — report it); a half/extended precision "
            f"operand is genuinely unserved.  Unset LORRAX_BANDS_GEMM_FFI "
            f"for this path — the XLA lowering accepts it, and explicit "
            f"requests are never silently downgraded.")

    def _right_none(o_l, psi_l):
        # (k, s, x, t, y) × (k, t, y, n) -> (k, s, x, n)
        if not use_ffi or not _ffi_dtypes_ok(o_l, psi_l):
            return jnp.einsum('ksxty,ktyn->ksxn', o_l, psi_l, optimize=True)
        k, s, x, t, y = o_l.shape
        n = psi_l.shape[3]
        c3 = _gemm_batch_ffi(o_l.reshape(k, s * x, t * y),
                             psi_l.reshape(k, t * y, n))
        return c3.reshape(k, s, x, n)

    def _right_leading(o_l, psi_l):
        # (e, k, s, x, t, y) × (k, t, y, n) -> (e, k, s, x, n); the k-only
        # ψ batch is broadcast over e by the handler's BA % BB rule.
        if not use_ffi or not _ffi_dtypes_ok(o_l, psi_l):
            return jnp.einsum('eksxty,ktyn->eksxn', o_l, psi_l,
                              optimize=True)
        e, k, s, x, t, y = o_l.shape
        n = psi_l.shape[3]
        c3 = _gemm_batch_ffi(o_l.reshape(e * k, s * x, t * y),
                             psi_l.reshape(k, t * y, n))
        return c3.reshape(e, k, s, x, n)

    def _right_minor(o_l, psi_l):
        # (k, s, x, t, y, e) × (k, t, y, n) -> (k, s, x, n, e).  XLA path
        # only (FFI refused above).
        return jnp.einsum('ksxtye,ktyn->ksxne', o_l, psi_l, optimize=True)

    _RIGHT = {"none": _right_none, "leading": _right_leading,
              "minor": _right_minor}

    def _right_maybe_split(o_l, psi_l, right_fn):
        """Policy 3: real O × complex ψ must never reach a mixed-dtype dot
        (XLA promotes the real operand to c128 — the ~400 MB convert copies
        + 2× flops the memo priced).  Split ψ into f64 parts, run pure-f64
        dgemms, recombine.  Genuinely complex O keeps ONE complex GEMM
        (f64-splitting it was measured as a regression — module docstring)."""
        if jnp.iscomplexobj(o_l) or not jnp.iscomplexobj(psi_l):
            return right_fn(o_l, psi_l)
        re = right_fn(o_l, jnp.real(psi_l))
        im = right_fn(o_l, jnp.imag(psi_l))
        return jax.lax.complex(re, im)

    # -- shard_map bodies ----------------------------------------------

    def _body_split_reim(psi_xr_local, o_local, psi_yn_local):
        # The two-channel (S_R, S_I) plan — op-for-op the historical
        # gw.ppm_tau_kernel._project_ri_local (subsumed here 2026-07-28):
        # shared f64 ψ extracts, two de-promoted real channels, stacked
        # collectives, complex left dots.
        psi_yn_re = jnp.real(psi_yn_local)
        psi_yn_im = jnp.imag(psi_yn_local)

        def _right(sigma_real_or_imag):
            re = _right_none(sigma_real_or_imag, psi_yn_re)
            im = _right_none(sigma_real_or_imag, psi_yn_im)
            return jax.lax.complex(re, im)

        right_re = _right(jnp.real(o_local))
        right_im = _right(jnp.imag(o_local))
        # ONE psum_scatter(y) for both channels; n is axis 4 of the stack:
        # (2, nk, s, μ_X_loc, n) → (2, nk, s, μ_X_loc, n/p_y)
        right_rs = jax.lax.psum_scatter(
            jnp.stack([right_re, right_im], axis=0), ax_y,
            scatter_dimension=4, tiled=True)

        def _left(right_rs_ch):
            return jnp.einsum(
                'kmsx,ksxn->kmn',
                jnp.conj(psi_xr_local), right_rs_ch, optimize=True)

        result_re = _left(right_rs[0])
        result_im = _left(right_rs[1])
        # ONE psum_scatter(x) for both channels; m is axis 2 of the stack:
        # (2, nk, m, n/p_y) → (2, nk, m/p_x, n/p_y)
        out = jax.lax.psum_scatter(
            jnp.stack([result_re, result_im], axis=0), ax_x,
            scatter_dimension=2, tiled=True)
        return (out[0].astype(jnp.complex128),
                out[1].astype(jnp.complex128))

    def _body_none(psi_xr_local, o_local, psi_yn_local):
        # Single-chain plan — op-for-op the historical
        # gw.ppm_tau_kernel._project_x_local when O is complex and
        # extra="none"; de-promoted (policy 3) when O is real.
        right = _right_maybe_split(o_local, psi_yn_local, _right_none)
        # ONE psum_scatter(y): (nk, s, μ_X_loc, n) → (nk, s, μ_X_loc, n/p_y)
        right_rs = jax.lax.psum_scatter(right, ax_y,
                                        scatter_dimension=3, tiled=True)
        result = jnp.einsum('kmsx,ksxn->kmn',
                            jnp.conj(psi_xr_local), right_rs, optimize=True)
        # ONE psum_scatter(x): (nk, m, n/p_y) → (nk, m/p_x, n/p_y)
        return jax.lax.psum_scatter(result, ax_x,
                                    scatter_dimension=1, tiled=True)

    def _body_leading(psi_xr_local, o_local, psi_yn_local):
        right = _right_maybe_split(o_local, psi_yn_local, _right_leading)
        # ONE stacked psum_scatter(y) for all E slices (policy 2):
        # (E, nk, s, μ_X_loc, n) → (E, nk, s, μ_X_loc, n/p_y)
        right_rs = jax.lax.psum_scatter(right, ax_y,
                                        scatter_dimension=4, tiled=True)
        result = jnp.einsum('kmsx,eksxn->ekmn',
                            jnp.conj(psi_xr_local), right_rs, optimize=True)
        # (E, nk, m, n/p_y) → (E, nk, m/p_x, n/p_y)
        return jax.lax.psum_scatter(result, ax_x,
                                    scatter_dimension=2, tiled=True)

    def _body_minor(psi_xr_local, o_local, psi_yn_local):
        right = _right_maybe_split(o_local, psi_yn_local, _right_minor)
        # (nk, s, μ_X_loc, n, E) → (nk, s, μ_X_loc, n/p_y, E)
        right_rs = jax.lax.psum_scatter(right, ax_y,
                                        scatter_dimension=3, tiled=True)
        result = jnp.einsum('kmsx,ksxne->kmne',
                            jnp.conj(psi_xr_local), right_rs, optimize=True)
        # (nk, m, n/p_y, E) → (nk, m/p_x, n/p_y, E)
        return jax.lax.psum_scatter(result, ax_x,
                                    scatter_dimension=1, tiled=True)

    # -- specs, guard, shard_map ---------------------------------------

    psi_left_spec = P(None, None, None, ax_x)
    psi_right_spec = P(None, None, ax_y, None)
    if extra == "leading":
        o_spec = P(None, None, None, ax_x, None, ax_y)
        out_spec = P(None, None, ax_x, ax_y)
        body = _body_leading
        o_axes = ("E", "nk", "s", "mu", "s2", "nu")
    elif extra == "minor":
        o_spec = P(None, None, ax_x, None, ax_y, None)
        out_spec = P(None, ax_x, ax_y, None)
        body = _body_minor
        o_axes = ("nk", "s", "mu", "s2", "nu", "E")
    else:
        o_spec = P(None, None, ax_x, None, ax_y)
        body = _body_split_reim if channels == "split_reim" else _body_none
        out_spec = ((P(None, ax_x, ax_y), P(None, ax_x, ax_y))
                    if channels == "split_reim" else P(None, ax_x, ax_y))
        o_axes = ("nk", "s", "mu", "s2", "nu")

    _sm = shard_map(body, mesh=mesh_xy,
                    in_specs=(psi_left_spec, o_spec, psi_right_spec),
                    out_specs=out_spec, check_rep=False)

    def _check_shapes(psi_left, o, psi_right):
        if o.ndim != len(o_axes):
            raise ValueError(
                f"contract_bands_block_reshard(extra={extra!r}): O must be "
                f"rank {len(o_axes)} {o_axes}, got shape {tuple(o.shape)}")
        off = 1 if extra == "leading" else 0
        nk, s, mu, s2, nu = (o.shape[off], o.shape[off + 1], o.shape[off + 2],
                             o.shape[off + 3], o.shape[off + 4])
        if psi_left.ndim != 4 or psi_right.ndim != 4:
            raise ValueError(
                f"psi_left/psi_right must be rank 4 (nk, m, s, mu) / "
                f"(nk, s', nu, n); got {tuple(psi_left.shape)} / "
                f"{tuple(psi_right.shape)}")
        if (psi_left.shape[0] != nk or psi_right.shape[0] != nk
                or psi_left.shape[2] != s or psi_left.shape[3] != mu
                or psi_right.shape[1] != s2 or psi_right.shape[2] != nu):
            raise ValueError(
                f"operand extents disagree with O {o_axes} = "
                f"{tuple(o.shape)}: psi_left (nk, m, s, mu) = "
                f"{tuple(psi_left.shape)}, psi_right (nk, s', nu, n) = "
                f"{tuple(psi_right.shape)}")
        if channels == "split_reim" and not jnp.iscomplexobj(o):
            raise TypeError(
                f"channels='split_reim' needs a complex O to split, got "
                f"{o.dtype}; a real O already IS one channel — use "
                f"channels='none' (the de-promoted chain applies "
                f"automatically).")
        m, n = int(psi_left.shape[1]), int(psi_right.shape[3])
        if m % p_x or n % p_y:
            raise ValueError(_DIVISIBILITY_MSG.format(
                m=m, n=n, px=p_x, py=p_y, ax_x=ax_x, ax_y=ax_y,
                hint=(("  " + divisibility_hint) if divisibility_hint
                      else "")))

    def _project(psi_left, o, psi_right):
        _check_shapes(psi_left, o, psi_right)
        return _sm(psi_left, o, psi_right)

    # Policy 4: create this mesh's MPI cliques on the MAIN thread now, so the
    # pool-worker collectives inside the returned kernel hit XLA's clique
    # cache instead of its MPI_Is_thread_main guard.  No-op off impl=mpi.
    warm_mesh_cliques(mesh_xy)

    return _project
