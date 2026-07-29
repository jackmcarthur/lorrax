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
4. **The impl=mpi world-collective-first warm-up contract**
   (memo Sec. 4.2, jobs 7878862/7878883): under
   ``JAX_CPU_COLLECTIVES_IMPLEMENTATION=mpi`` the XLA CPU mpi-collectives
   runtime creates GROUPED cliques lazily and its
   ``MPI_Is_thread_main`` check passes ONLY after a WORLD-clique
   collective has first-touched the runtime; a standalone consumer whose
   FIRST collective is a grouped psum_scatter dies deterministically with
   ``UNKNOWN: MPI: Communicator requested from a thread ...``.
   Async-dispatch off does NOT avoid it — the contract is ORDER, not
   asynchrony.  Production runs satisfy it accidentally (early sync
   barriers); tests, microbenches and future BSE drivers do not.  The
   factory therefore calls :func:`ensure_grouped_collectives_ready`
   before returning — every consumer of this primitive inherits the
   warm-up instead of rediscovering the failure.
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
host FFI handler (``lorrax_mklblas_gemm_batch``, src/ffi/mklblas/cpp —
{d,s,z,c}gemm dispatch on buffer dtype (all four BLAS
precisions), batched ``cblas_?gemm_batch`` entry
when the BLAS has it, plain cblas GEMM loop otherwise; works in principle
with Intel MKL or Cray LibSci, tested with Intel only so far;
vendor-internal threading under the workstream-AW MklThreadScope
pattern).  Everything else — channel algebra, collectives, the small left
dots (1.6e-3 of the right's flops, measured) — is untouched.

Default is AUTO (owner order 2026-07-29, doctrine #8: capability
detection, not policy): unset / ``auto`` turns the FFI body ON when the
platform is CPU AND the handler resolves in the host .so — announced once
on rank 0 — and quietly keeps the native lowering everywhere the dial
cannot apply (non-CPU platform: cuBLAS native is optimal, silent BY
DESIGN; ``extra="minor"``, which is structural — the contracted axis is
not GEMM-reachable there).  Since 2026-07-29 the handler serves ALL FOUR
BLAS precisions (f64/f32/c128/c64), so the BSE fp32-GMRES complex64 path
rides it too and is no longer a capability boundary.  ``LORRAX_BANDS_GEMM_FFI=0`` disables.  An EXPLICIT ``=1`` keeps
the announce-or-REFUSE semantics: it REFUSES on a non-CPU mesh, refuses
when the host .so lacks the handler (quoting the probe reason), refuses
``extra="minor"`` and refuses unsupported dtypes — explicit requests are
never silently downgraded.  ``input_output_aliases``: deliberately NONE —
a GEMM output (B, M, N) never matches an operand buffer shape, so no
alias is legal (contrast the in-place FFT handlers).  Read at FACTORY
time: kernel caches must key on :func:`bands_gemm_ffi_enabled`
(ppm_tau_kernel does).

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

import os
from typing import Callable

import jax
import jax.numpy as jnp
from jax.sharding import Mesh, PartitionSpec as P

__all__ = [
    "contract_bands_block_reshard",
    "bands_gemm_ffi_enabled",
    "bands_gemm_ffi_mode",
    "ensure_grouped_collectives_ready",
]


# ---------------------------------------------------------------------------
# impl=mpi world-collective-first warm-up (policy 4)
# ---------------------------------------------------------------------------

_world_warmed = False


def ensure_grouped_collectives_ready(*, print_fn=print) -> bool:
    """Satisfy the XLA mpi-collectives world-collective-first init contract.

    Under ``JAX_CPU_COLLECTIVES_IMPLEMENTATION=mpi`` a GROUPED clique
    (any psum_scatter over a mesh sub-axis) can only be created after a
    WORLD-clique collective has first-touched the collectives runtime —
    otherwise every rank whose first collective is grouped dies with
    ``UNKNOWN: MPI: Communicator requested from a thread that is not the
    one MPI was initialized from``  (memo Sec. 4.2; resolved empirically
    by the 7878883 probe: world-psum-first fixes it, async-dispatch-off
    does not).  This helper issues one world-spanning barrier collective
    (``common.collectives.barrier`` — a psum over ALL global devices)
    exactly once per process, only when the mpi implementation is active
    in a multi-process run.  Idempotent and cheap when already warm
    (~60 ms measured); a no-op on gloo / single-process / non-CPU stacks.

    MUST be called synchronously on every rank (it is a collective).  The
    :func:`contract_bands_block_reshard` factory calls it, and factories
    are invoked synchronously on all ranks by construction (kernel-build
    time); standalone consumers (microbenches, drivers) may call it
    directly before their first grouped collective.

    Returns True when a warm-up collective was actually executed.
    """
    global _world_warmed
    if _world_warmed:
        return False
    impl = os.environ.get(
        "JAX_CPU_COLLECTIVES_IMPLEMENTATION", "").strip().lower()
    if impl != "mpi":
        return False
    from common.collectives import barrier, process_count
    if process_count() <= 1:
        return False
    # A real world-clique collective through the active runtime.  Marking
    # the flag BEFORE the barrier would skip the warm-up forever if the
    # barrier itself raised; barrier() is fatal-on-failure by design, so
    # ordering after is safe and re-entry cannot happen on a dead rank.
    barrier("contract_bands.world_collective_warmup", print_fn=print_fn)
    _world_warmed = True
    return True


# ---------------------------------------------------------------------------
# Gated vendor-BLAS GEMM FFI body (CPU only; AUTO default since 2026-07-29)
# ---------------------------------------------------------------------------

_BANDS_GEMM_TARGET = "lorrax_mklblas_gemm_batch"
_bands_gemm_announced: set = set()
_bands_gemm_auto_verdict: bool | None = None  # memoized capability probe


def _rank0() -> bool:
    try:
        return jax.process_index() == 0
    except Exception:                                     # noqa: BLE001
        return True


def bands_gemm_ffi_mode() -> str:
    """The LORRAX_BANDS_GEMM_FFI grammar: ``"on"`` | ``"off"`` | ``"auto"``.

    ``auto`` (the DEFAULT since the 2026-07-29 owner order — unset or
    ``auto``) is capability detection, not policy (env-vars doctrine #8):
    it turns the FFI GEMM body ON only where it is known-optimal and
    available (CPU platform + handler in the host .so) and quietly leaves
    the native lowering everywhere else.  Explicit ``1`` keeps the
    announce-or-REFUSE semantics; explicit ``0`` disables.  Unrecognized
    values announce once and take the conservative direction: ``off``."""
    v = os.environ.get("LORRAX_BANDS_GEMM_FFI", "").strip().lower()
    if v in ("", "auto"):
        return "auto"
    if v in ("0", "off", "false", "no"):
        return "off"
    if v in ("1", "on", "true", "yes"):
        return "on"
    if "grammar" not in _bands_gemm_announced:
        _bands_gemm_announced.add("grammar")
        print(f"*** LORRAX_BANDS_GEMM_FFI={v!r} is not a recognized value "
              f"(accepted: auto/unset, 0/off/false/no, 1/on/true/yes).  "
              f"Treating as OFF (default XLA GEMM lowering). ***", flush=True)
    return "off"


def _bands_gemm_auto_enabled() -> bool:
    """The AUTO resolution (owner order 2026-07-29): ON iff the platform
    is CPU AND the GEMM handler resolves in the host .so.  Decision
    announced ONCE on rank 0 either way (the router prints its decision);
    on a non-CPU platform auto is OFF silently BY DESIGN — XLA:GPU's dot
    lowering already dispatches cuBLAS, which is optimal there.

    Platform is read from ``JAX_PLATFORMS`` (``ffi_loader.platform_from_env``
    — never initializes the JAX backend, so cache-key calls before
    ``jax.distributed.initialize`` stay safe).  The production CPU harnesses
    export ``JAX_PLATFORMS=cpu``, and ``runtime.bootstrap()``'s GPU-less
    downgrade (``fallback_to_cpu_if_no_gpu_backend``) FORCES it to ``cpu``,
    so those runs resolve correctly.  The read is lexical though: an unset
    variable resolves CUDA-first, and so does a run that reaches a CPU mesh
    while ``JAX_PLATFORMS`` still reads ``cuda,cpu`` (``set_default_env``'s
    gpu default, if jax resolved cpu without raising).  Both give auto-OFF
    — the safe direction: the XLA lowering, no error, just no speedup.
    Fix either by exporting ``JAX_PLATFORMS=cpu`` or ``=1`` explicitly.
    """
    global _bands_gemm_auto_verdict
    if _bands_gemm_auto_verdict is not None:
        return _bands_gemm_auto_verdict
    from ffi.common import ffi_loader
    if ffi_loader.platform_from_env(default="CUDA") != "cpu":
        # Non-CPU platform: silently-by-design OFF (cuBLAS native optimal).
        _bands_gemm_auto_verdict = False
        return False
    ok, reason = ffi_loader.probe_target(_BANDS_GEMM_TARGET, "cpu")
    _bands_gemm_auto_verdict = bool(ok)
    if "auto" not in _bands_gemm_announced:
        _bands_gemm_announced.add("auto")
        if _rank0():
            if ok:
                print(f"[bands_gemm] AUTO-ON: CPU platform and FFI target "
                      f"{_BANDS_GEMM_TARGET!r} resolves in the host .so -> "
                      f"contract_bands right-GEMMs at vendor-BLAS rate "
                      f"(1.6-1.9x over the XLA:CPU Eigen dots at full "
                      f"threads, jobs 7879008/7879010).  "
                      f"LORRAX_BANDS_GEMM_FFI=0 disables.", flush=True)
            else:
                print(f"[bands_gemm] auto: FFI target {_BANDS_GEMM_TARGET!r} "
                      f"unavailable -> default XLA GEMM lowering "
                      f"({reason})", flush=True)
    return _bands_gemm_auto_verdict


def bands_gemm_ffi_enabled() -> bool:
    """True when the primitive's LARGE right contraction routes through the
    vendor-BLAS GEMM host FFI handler.  DEFAULT is AUTO (2026-07-29 owner
    order): ON when the platform is CPU and the handler resolves in the
    host .so (announced once), OFF otherwise — see
    :func:`bands_gemm_ffi_mode`.  Read at FACTORY time — kernel caches
    must key on this (ppm_tau_kernel's pipeline/cache keys do)."""
    mode = bands_gemm_ffi_mode()
    if mode == "off":
        return False
    if mode == "on":
        return True
    return _bands_gemm_auto_enabled()


def _require_bands_gemm_ffi(mesh: Mesh) -> None:
    """Announce-or-refuse for the explicitly requested FFI GEMM body."""
    plat = mesh.devices.flat[0].platform
    if plat != "cpu":
        raise RuntimeError(
            f"LORRAX_BANDS_GEMM_FFI requested the MKL batched-GEMM host "
            f"backend, but the mesh devices are {plat!r}.  This dial is "
            f"CPU-only BY DESIGN: the native XLA dot lowering on CUDA "
            f"already dispatches cuBLAS (optimal there) and is deliberately "
            f"untouched.  Unset LORRAX_BANDS_GEMM_FFI on non-CPU meshes — "
            f"explicit requests are never silently downgraded.")
    from ffi.common import ffi_loader
    ok, reason = ffi_loader.probe_target(_BANDS_GEMM_TARGET, "cpu")
    if not ok:
        raise RuntimeError(
            f"LORRAX_BANDS_GEMM_FFI requested the MKL batched-GEMM host "
            f"backend, but FFI target {_BANDS_GEMM_TARGET!r} is unusable: "
            f"{reason}")
    if _BANDS_GEMM_TARGET not in _bands_gemm_announced:
        _bands_gemm_announced.add(_BANDS_GEMM_TARGET)
        if _rank0():
            print(f"[bands_gemm] contract_bands right-GEMMs -> vendor-BLAS "
                  f"GEMM host FFI handler ({_BANDS_GEMM_TARGET}): "
                  f"dgemm/zgemm at BLAS rate (memo Sec. 4.4 exit (b)); "
                  f"collectives, channel algebra and left dots untouched.",
                  flush=True)


def _gemm_batch_ffi(a3, b3):
    """C[i] = A[i] @ B[i % BB] for A (BA, M, K), B (BB, K, N), BA % BB == 0.

    Row-major NN batched GEMM through the host handler; dtype
    (f64/f32/c128/c64)
    is dispatched inside the .so from the buffer element type.  The
    broadcast rule (B cycling with period BB) is what lets one handler
    serve both the plain per-k batch (BA == BB == nk) and the
    extra-stacked batch (BA == E·nk against the k-only ψ, extra axis
    OUTERMOST).  No input_output_aliases — a (BA, M, N) GEMM output can
    never legally alias a (BA, M, K)/(BB, K, N) operand buffer.
    """
    ba, m, k = (int(d) for d in a3.shape)
    bb, k2, n = (int(d) for d in b3.shape)
    if k != k2 or (bb and ba % bb):
        raise ValueError(
            f"bands_gemm: incompatible batch shapes A{tuple(a3.shape)} vs "
            f"B{tuple(b3.shape)} (need A.K == B.K and BA % BB == 0).")
    if a3.dtype != b3.dtype:
        raise TypeError(
            f"bands_gemm: mixed operand dtypes {a3.dtype} vs {b3.dtype} — "
            f"the de-promotion policy should have split them upstream.")
    out_t = jax.ShapeDtypeStruct((ba, m, n), a3.dtype)
    return jax.ffi.ffi_call(_BANDS_GEMM_TARGET, out_t)(a3, b3)


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
        # keep the native lowering where the dial cannot apply.  Non-CPU
        # mesh: silent OFF by design (XLA:GPU's dot lowering already hits
        # cuBLAS — optimal); extra='minor': the contracted axis is not
        # GEMM-reachable (see the explicit-mode refusal above), so the
        # minor order keeps the XLA plan.  The probe + AUTO-ON announce
        # already ran inside bands_gemm_ffi_enabled().
        plat = mesh_xy.devices.flat[0].platform
        if plat != "cpu" or extra == "minor":
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

    # Policy 4: every consumer inherits the warm-up (no-op off-mpi).
    ensure_grouped_collectives_ready()

    return _project
