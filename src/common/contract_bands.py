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

That is the ``layout="legacy"`` (default) body, byte-identical to the
code this module shipped before the face carrier existed.
``layout="face"`` (the band-distributed ψ carrier every GW stage holds,
``gw.wavefunction_bundle``) solves the SAME projection with every operand
at 1/P: the operator stays on its tiles, ψ_l becomes a resident 1/P slab,
ψ_r streams through in band chunks sized against one operator tile, and
two psum_scatters (T's ν sum, then the band block) finish it
(:func:`_face_project_kernel`, which prices its collectives and peak).
``layout="axis"`` operands already carry every band, so each rank holds a
whole ``(μ_x, ν_y)`` slab of the contraction: :func:`_axis_project_kernel`
contracts it locally and reduces the ``(nb, nb)`` partial ONCE with the
slab-contraction primitive (:func:`reduce_scatter_to_band_block`).  The
legacy body is not a GW route.

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
the Σ project_rs row, running ~2.1× below the node's BLAS roofline
through XLA:CPU's Eigen dots (295 GF/s promoted zgemm / ~172 GF/s split
dgemm vs 1263 GF/s MKL as first written, but that RATIO is wrong — the
audit recomputes Eigen at ~631 GF/s, FFI_EVIDENCE_AUDIT.md G1/G18,
corrected 2026-08-11; the wall-clocks stand.  memo Sec. 4.4/4.5; the bare Eigen dot saturates
1.6–1.9× below vendor BLAS at full threads — jobs 7879008/7879010).
The dial routes ONLY that right contraction through the vendor-BLAS GEMM
host FFI handler; everything else — channel algebra, collectives, the
small left dots (1.6e-3 of the right's flops, measured) — is untouched.

The dial ITSELF is a microservice: ``ffi.gemm`` owns its grammar,
platform resolution, capability probe, announcements, refusals and the
``ffi_call`` (handler ``lorrax_mklblas_gemm_batch``,
src/ffi/cpp/cblas).  ``docs/dev/vendor_gemm_service.md`` is its
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

import functools
from typing import Callable

import jax
import numpy as np
import jax.numpy as jnp
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

from common.collectives import warm_mesh_cliques
from runtime.padding import authenticate_padded_axis, padded_axis

__all__ = [
    "contract_bands_block_reshard",
    "BandProjector",
    "face_projection_chunk",
    "bands_gemm_ffi_enabled",
    "bands_gemm_ffi_mode",
    "merge_spin_centroid",
    "split_spin_centroid",
    "bands_to_contraction_slabs",
    "reduce_scatter_to_band_block",
]


# ---------------------------------------------------------------------------
# THE SLAB-CONTRACTION PRIMITIVE (rules R1/R3,
# reports/gwjax_scaling_levers_2026-09-23): when a band-pair output is small
# against the axis it contracts (⟨m|O|n⟩ over G, Σ_mn over μν), split the
# CONTRACTION axis over the whole mesh, contract each rank's slab locally,
# and reduce the small (…, m, n) partial ONCE into the (m_x, n_y) block.
# Both helpers run INSIDE a ``shard_map`` over ``axes``; the linear rank is
# ``x·p_y + y`` (the ``band_sphere_spec`` block order), and every
# collective here is one call per invocation however large the stack.
# First consumer: ``common.mtxel_sweep.sweep_matrix_elements``.
# ---------------------------------------------------------------------------

def bands_to_contraction_slabs(a, *, band_axis: int, slab_axis: int,
                               carrier: int, axes=("x", "y")):
    """Band-split → contraction-split, one all-to-all over ``axes``.

    ``a`` holds this rank's ``nb/P`` bands (block ``x·p_y + y``) with the
    WHOLE contraction axis at ``slab_axis``; the result holds EVERY band in
    global order with this rank's ``carrier/P`` slab of that axis.  The
    contraction axis is zero-padded to ``carrier`` (a multiple of P, from
    ``runtime.padding.padded_axis``) first, so pad columns are exact zeros.

    The operand is pinned row-major: an all-to-all wants its split axis
    major, and left free, layout assignment can satisfy that on a scan's
    LOOP OPERAND and hoist a transposed copy of the whole resident array
    out of the loop (+ψ/P per rank, ``runs/runtime/density_scan_20260923``
    legs b02/b05).  Pinned, the transpose is one operand wide.
    """
    from jax.experimental.layout import Layout, with_layout_constraint

    a = with_layout_constraint(a, Layout(major_to_minor=tuple(range(a.ndim))))
    pad = [(0, 0)] * a.ndim
    pad[slab_axis] = (0, int(carrier) - int(a.shape[slab_axis]))
    return jax.lax.all_to_all(jnp.pad(a, pad), axes, split_axis=slab_axis,
                              concat_axis=band_axis, tiled=True)


def reduce_scatter_to_band_block(part, *, px: int, py: int,
                                 axes=("x", "y"), row_axis: int = -2,
                                 col_axis: int = -1):
    """Σ over the mesh of a ``(…, m, …, n, …)`` partial → this rank's block.

    Every rank holds a partial of the WHOLE ``(m, n)`` matrix (its slab's
    share of the contraction); one reduce-scatter over ``axes`` sums them
    and delivers rows ``x·m/p_x + [0, m/p_x)`` and columns
    ``y·n/p_y + [0, n/p_y)`` — the block whose spec carries 'x' at
    ``row_axis`` and 'y' at ``col_axis`` (default the trailing pair).  ``m``
    and ``n`` must divide ``p_x`` and ``p_y``.

    THE ONE REPLICATED OBJECT, priced: the partial is ``prod(lead)·m·n``
    elements per rank.  Against a 2-D band split that gathers operand
    panels of ``(1 + c)·N/√P`` per rank, it wins while ``nb·√P < N``, N the
    contracted extent per band (``ns·N_G`` for a matrix element).
    """
    ra, ca = row_axis % part.ndim, col_axis % part.ndim
    if not ra < ca:
        raise ValueError("reduce_scatter_to_band_block: row_axis must "
                         "precede col_axis")
    m, n = int(part.shape[ra]), int(part.shape[ca])
    shape = list(part.shape)
    shape[ca:ca + 1] = [py, n // py]
    shape[ra:ra + 1] = [px, m // px]        # ra < ca: split the later first
    r = jnp.moveaxis(part.reshape(shape), (ra, ca + 1), (0, 1))
    r = r.reshape(px * py, *r.shape[2:])
    return jax.lax.psum_scatter(r, axes, scatter_dimension=0, tiled=False)


# ---------------------------------------------------------------------------
# GEMM-seam (s, mu) merge — the face GEMM operands' one (s, mu) order
# (gw.greens_function_kernel's face G builder, wavefunction_bundle's
# rotations).
# ---------------------------------------------------------------------------

def merge_spin_centroid(x, spin_axis: int, centroid_axis: int):
    """Merge an adjacent (spin, centroid) axis pair into ONE axis, for a
    cuBLASMp N,N GEMM operand at the two-face carrier's (s,μ) seam.

    ``spin_axis`` must hold a REPLICATED axis (``s``/``ns`` — size 1 or 2 in
    every deck this has run against) and ``centroid_axis`` the adjacent
    MESH-SHARDED one (``mu``/``nu``); ``centroid_axis == spin_axis + 1`` is
    required (every ψ_nmu/ψ_mun/O field in this codebase stores them that
    way — ``wavefunction_bundle.PSI_MUN_SPEC``/``PSI_NMU_SPEC``,
    ``contract_bands``'s own ``O`` operand).

    **The merge ORDER is load-bearing, not cosmetic — MEASURED, not a
    style choice.**  A plain ``reshape`` merging the pair in STORAGE order
    (spin outer/major, centroid inner/minor — what a naive "flatten (s,mu)"
    reading of the audit report would do) is not a free reshape: traced on
    an emulated 2x2 mesh, it lowers to a genuine ``all-to-all`` collective
    for ns > 1 (verified in a throwaway probe kept out of the tree; the
    HLO contained ``all-to-all(...channel_id=...)`` with no
    ``with_sharding_constraint`` able to avoid it).  Putting the SHARDED
    axis major and the replicated spin axis minor — this function's
    ``jnp.swapaxes`` THEN ``reshape`` — is the free direction: zero
    collectives in the compiled HLO, bit-exact against plain NumPy
    transpose+reshape, and every rank's local shard equals the slice a
    genuinely chunked ``P(...,'x'|'y',...)`` array would hold at that
    merged position.  :func:`split_spin_centroid` is the exact inverse.

    Returns the array with the two axes replaced by one of size
    ``spin_size * centroid_size``, centroid-major (``merged = c*ns + s``).
    """
    if centroid_axis != spin_axis + 1:
        raise ValueError(
            f"merge_spin_centroid: centroid_axis={centroid_axis} must be "
            f"spin_axis={spin_axis} + 1 (adjacent, centroid immediately "
            "after spin) — every face-layout field in this tree stores "
            "them that way; a non-adjacent pair needs its own transpose "
            "at the call site before this helper, not a silent extra one "
            "hidden in here.")
    xt = jnp.swapaxes(x, spin_axis, centroid_axis)
    shape = xt.shape
    merged = (shape[:spin_axis] + (shape[spin_axis] * shape[spin_axis + 1],)
             + shape[spin_axis + 2:])
    return xt.reshape(merged)


def split_spin_centroid(x, axis: int, spin_size: int, centroid_size: int):
    """Inverse of :func:`merge_spin_centroid`: split the merged axis at
    ``axis`` (size ``spin_size * centroid_size``, centroid-major) back into
    two axes ``(spin, centroid)`` at ``(axis, axis + 1)`` — restoring the
    bundle's own storage order (spin outer, centroid inner)."""
    shape = x.shape
    unmerged = shape[:axis] + (centroid_size, spin_size) + shape[axis + 1:]
    y = x.reshape(unmerged)
    return jnp.swapaxes(y, axis, axis + 1)


# ---------------------------------------------------------------------------
# Gated vendor-BLAS GEMM FFI body — the SERVICE lives in ``ffi.gemm``
# ---------------------------------------------------------------------------
# The dial's grammar, platform resolution, capability probe, announcements
# and refusals are the microservice's (``src/ffi/gemm.py``, on the
# shared ``ffi.gate.Gate``); its ``ffi_call`` is there too.  This module keeps
# only the two things that
# are genuinely ITS policy and cannot live in a GEMM service: WHICH
# contraction is routed (the large right one, never the left dots), and the
# ``extra="minor"`` structural exclusion, which is a fact about this
# primitive's operand layout rather than about BLAS.  Contract:
# ``docs/dev/vendor_gemm_service.md``; gate doctrine:
# ``docs/dev/ffi_gate_contract.md``.
from ffi.gemm import (                                       # noqa: E402
    GATE as _BANDS_GEMM_GATE,
    gemm_batch as _gemm_batch_ffi,
    require_gemm_ffi as _require_bands_gemm_ffi,
)


def bands_gemm_ffi_mode() -> str:
    """The LORRAX_BANDS_GEMM_FFI grammar: ``"on"`` | ``"off"``
    (delegates to :data:`ffi.gemm.GATE`)."""
    return _BANDS_GEMM_GATE.mode()


def bands_gemm_ffi_enabled() -> bool:
    """True when the primitive's LARGE right contraction routes through the
    vendor-BLAS GEMM host FFI handler.  DEFAULT is ON — the FFI layer is
    REQUIRED (decisions.md 2026-08-01): a missing handler refuses at
    startup (``Gate.enforce``) instead of demoting, and ``=0`` is an
    announced, uncertified debug opt-out onto the retained XLA einsum arm
    (retained because the ``extra='minor'`` order structurally cannot ride
    a batched GEMM — see below — so the native arm is not a deletable
    duplicate here).  Read at FACTORY time — kernel caches must key on
    this (ppm_tau_kernel's pipeline/cache keys do); the read is env-only
    and never initializes the JAX backend (gate contract, tier 1)."""
    return _BANDS_GEMM_GATE.enabled()


# ---------------------------------------------------------------------------
# The primitive
# ---------------------------------------------------------------------------

#: ``(nk, nb_full, n_rmu, nspinor)`` — the four static ints
#: :func:`contract_bands_block_reshard`'s face and axis paths need to fix
#: their local tile shapes and the face path's band-chunk count at
#: construction (the legacy shard_map body is shape-polymorphic).  A plain
#: 4-tuple rather than a new dataclass: this is the only site that reads it,
#: and every field already has a home name elsewhere
#: (``Wavefunctions.slices.nb_full``, ``psi_mun.shape``).
FaceProjectShape = tuple


def face_projection_chunk(*, nk, nb, ns, mu_left, mu_right, p, channels=1,
                          itemsize=16):
    """Local band columns ``w`` per chunk of the face projector's stream.

    ``mu_left/p`` is the operator block's local μ rows (an x block's
    ``xn·bx``).  The stream's transients
    per rank, for ``w`` local columns (``p·w`` global): the gathered ψ_r
    chunk ``nk·ns·(μ_r/p)·p·w`` (twice while a prefetch is in flight), and
    per channel the partial ``T`` ``nk·ns·(μ_l/p)·p·w``, its scattered slab
    ``nk·ns·⌈μ_l/p⌉_p/p·p·w`` and the band partial ``nk·nb·p·w``.  They are
    admitted against ONE local operator tile ``nk·ns²·(μ_l/p)·(μ_r/p)`` —
    the block being projected, so the stream never more than doubles its
    operand's footprint and falls as 1/P.  The largest divisor of ``nb/p``
    that fits wins (one chunk, ``w = nb/p``, whenever everything fits);
    ``w = 1`` is the floor.
    """
    b_loc = nb // p
    mul, mur = mu_left // p, mu_right // p
    q = padded_axis(mul, int(p), name="face slab centroid piece").carrier // p
    tile = itemsize * nk * ns * mul * ns * mur
    for w in sorted((d for d in range(1, b_loc + 1) if b_loc % d == 0),
                    reverse=True):
        prefetch = 2 if w < b_loc else 1
        cols = p * w
        need = itemsize * nk * cols * (
            prefetch * ns * mur + channels * (ns * mul + ns * q + nb))
        if need <= tile:
            return w
    return 1


def face_row_blocks(mu_local, p, n_blocks):
    """The x blocks ``(x0, bx, xs, xn)`` an operator's local μ rows are stored and projected in.

    The face stream's T reduce-scatters its ``μ_x`` rows (padded to ``p·q``,
    ``q = ⌈μ_x/p⌉``) into ``p`` slab pieces of ``q``; block ``i`` takes rows
    ``[x0_i, x0_i + bx_i)`` of every piece (stride ``xs = q``, ``xn = p``
    pieces), so its T scatters onto ``slab[..., x0_i:x0_i + bx_i]`` and the
    blocks together move and multiply what the whole operator does.  The
    ``bx_i`` split ``q`` into ``n_blocks`` near-equal ranges (wider first).
    Rows ``>= μ_x`` are zero padding (``ffi.fft.x_block_rows``).
    """
    mu_local, p = int(mu_local), int(p)
    q = padded_axis(mu_local, p, name="face slab centroid piece").carrier // p
    n = max(1, min(int(n_blocks), q))
    base, extra = divmod(q, n)
    widths = [base + 1] * extra + [base] * (n - extra)
    starts = np.cumsum([0] + widths[:-1])
    return tuple((int(x0), int(bx), q, p) for x0, bx in zip(starts, widths))


class BandProjector:
    """``Σ[m,n] = Σ conj(ψ_l)[m,s,μ] O[s,μ,s',ν] ψ_r[s',ν,n]`` in three phases.

    ``prepare(psi_left, psi_right) -> faces`` orients the ψ operands once;
    ``accumulate(faces, O, rows=None, acc=None) -> acc`` adds one operator x
    block's contribution to a rank-local ``(nb, nb)`` partial: ``O`` holds
    the block's local μ rows (``rows = (x0, bx, xs, xn)`` from
    ``row_blocks``, :func:`face_row_blocks`; ``None``: the whole operator),
    every spin and every ν; ``finish(acc)`` reduces it ONCE into
    ``(nk, m_X, n_Y)``.  A caller that projects an operator in x blocks
    (``gw.ppm_tau_kernel``) prepares once and reduces once, however many
    blocks.  The partial is
    ``16·nk·nb²`` bytes per rank (per channel), P-independent, carried as a
    ``(nch, P·nk, nb, nb)`` array at ``P(None, (ax_x, ax_y))``.

    Calling the projector, ``project(psi_left, O, psi_right)``, is all three
    for a whole operator.  ``channels="split_reim"`` returns ``(S_R, S_I)``.
    """

    def __init__(self, *, mesh, axes, expected, in_specs, channels, prepare,
                 accumulate, finish, row_blocks):
        self.mesh, self.axes, self.channels = mesh, axes, channels
        self._expected, self._in_specs = expected, in_specs
        self.prepare, self._accumulate, self._finish = (
            prepare, accumulate, finish)
        self.row_blocks = row_blocks

    def accumulate(self, faces, O, rows=None, acc=None):
        rows = None if rows is None else tuple(int(v) for v in rows)
        return self._accumulate(faces, O, acc, rows=rows)

    def finish(self, acc):
        out = self._finish(acc)
        return out[0] if self.channels == "none" else (out[0], out[1])

    def __call__(self, psi_left, O, psi_right):
        got = (tuple(psi_left.shape), tuple(O.shape), tuple(psi_right.shape))
        if got != self._expected:
            raise ValueError(
                "contract_bands_block_reshard: endpoint shapes "
                f"{got} do not match planned {self._expected}")
        # Same contract as a planned GEMM: a concrete operand must already
        # sit in its spec (a tracer's layout belongs to the enclosing jit),
        # never an implicit (μ, n)-class reshard.
        for name, x, spec in zip(("psi_left", "O", "psi_right"),
                                 (psi_left, O, psi_right), self._in_specs):
            have = getattr(x, "sharding", None)
            if (not isinstance(x, jax.core.Tracer) and have is not None
                    and not have.is_equivalent_to(
                        NamedSharding(self.mesh, spec), x.ndim)):
                raise ValueError(
                    f"contract_bands_block_reshard: {name} must already be "
                    f"sharded {spec}; refusing an implicit reshard of a "
                    f"{tuple(x.shape)} array.  Got {have!r}.")
        return self.finish(self.accumulate(self.prepare(psi_left, psi_right),
                                           O))


def _projector_shapes(mesh_xy, face_shape, axes, channels, right_face_shape,
                      band_extent, row_block, *, square):
    """Validated static extents shared by the face and axis projectors."""
    ax_x, ax_y = axes
    px, py = int(mesh_xy.shape[ax_x]), int(mesh_xy.shape[ax_y])
    if square and px != py:
        raise ValueError(
            "contract_bands_block_reshard(layout='face'): the face projector "
            f"needs a square mesh, got {dict(mesh_xy.shape)}")
    if channels not in ("none", "split_reim"):
        raise ValueError(
            f"contract_bands_block_reshard: channels must be 'none' or "
            f"'split_reim', got {channels!r}")
    nk, nb_full, mu_l, ns = (int(v) for v in face_shape)
    right_face_shape = face_shape if right_face_shape is None \
        else right_face_shape
    if (int(right_face_shape[0]), int(right_face_shape[1]),
            int(right_face_shape[3])) != (nk, nb_full, ns):
        raise ValueError(
            "contract_bands_block_reshard: left/right face shapes must share "
            f"(nk, nb_full, nspinor); got {tuple(face_shape)} and "
            f"{tuple(right_face_shape)}")
    mu_r = int(right_face_shape[2])
    nb = nb_full if band_extent is None else int(band_extent)
    band_spec = P(None, ax_x, ax_y)                  # Σ (nk, m_X, n_Y)
    o_spec = P(None, None, ax_x, None, ax_y)         # O (nk, s, μ_X, s', ν_Y)
    authenticate_padded_axis(
        nb, nb, mesh_xy, specs=((band_spec, 1), (band_spec, 2)),
        name="contract_bands_block_reshard: projected band extent")
    authenticate_padded_axis(
        mu_l, mu_l, mesh_xy, spec=o_spec, axis=2,
        name="contract_bands_block_reshard: left centroid extent")
    authenticate_padded_axis(
        mu_r, mu_r, mesh_xy, spec=o_spec, axis=4,
        name="contract_bands_block_reshard: right centroid extent")
    rb = mu_l // px if row_block is None else int(row_block)
    if rb < 1:
        raise ValueError(
            f"contract_bands_block_reshard: row_block {rb} must be >= 1")
    return nk, nb, ns, rb, mu_l, mu_r, px, py


def _band_block_finish(mesh_xy, axes, px, py, nch):
    """``acc`` ``(nch, P·nk, nb, nb)`` partials → ``(nch, nk, m_X, n_Y)``."""
    from common.shard_map import shard_map

    ax_x, ax_y = axes
    return jax.jit(shard_map(
        lambda acc: reduce_scatter_to_band_block(
            acc, px=px, py=py, axes=axes, row_axis=2, col_axis=3),
        mesh=mesh_xy, in_specs=P(None, axes), out_specs=P(None, None, ax_x, ax_y),
        check_vma=False))


def _face_project_kernel(mesh_xy: Mesh, face_shape, axes, *,
                         channels: str = "none", right_face_shape=None,
                         band_extent=None, row_block=None):
    """The face-layout Σ projector: the operator stays, ψ streams in 1/P tiles.

        Σ[m,n] = Σ_{s,μ} conj(ψ_l)[m,s,μ] T[s,μ,n],
        T[s,μ,n] = Σ_{s',ν} O[s,μ,s',ν] ψ_r[s',ν,n]

    Operands, all 1/P on the square ``p×p`` mesh: ``ψ_l`` = psi_nmu
    ``(nk, m_X, s, μ_Y)``, ``O`` ``(nk, s, μ_X, s', ν_Y)``, ``ψ_r`` = psi_mun
    ``(nk, s', ν_X, n_Y)``; output ``(nk, m_X, n_Y)``.

    ``prepare``: each ψ tile goes to its transpose partner (rank (x,y) ↔
    (y,x), one ``ppermute`` of a 1/P tile), so rank (x,y) holds ψ_l(m_y,
    μ_x) and ψ_r(ν_y, n_x) — the centroid blocks of its own operator tile;
    ψ_l's μ_x block then splits over 'y' while its bands gather (one
    ``all_to_all``): the slab conj ψ_l(all m, μ_x piece y), 1/P.

    ``accumulate``, per band chunk (``p·w`` global columns, the next one
    gathered while this one multiplies): ψ_r's chunk gathers over 'x' to
    ``(ν_y, p·w)``; ``T = O_local·ψ_r`` is the local ZGEMM (the flops of
    the whole projection); one ``psum_scatter`` over 'y' completes the ν
    sum onto the slab's μ piece; ``slab·T`` adds the rank's share of the
    chunk's ``(nb, p·w)`` columns to the partial.  ``finish``: one
    band-block reduce-scatter.

    ``O`` never moves.  Per rank, a whole-spin projection moves
    ``16·nk·[nb·ns·(μ_l+μ_r)/p + nb² + 3·nb·ns·μ/P]`` bytes of collectives
    (``prepare`` is the last term); a stationary-output GEMM (SUMMA,
    cuBLASMp) would add ``16·nk·ns²·μ_l·μ_r/p``, √P operator tiles.  The
    stream's transients are admitted against one local operator tile
    (:func:`face_projection_chunk`) and fall as 1/P; the slab is
    ``16·nk·nb·ns·μ_l/P``; the partial ``16·nk·nb²``.  T's ν sum is reduced
    before the ψ_l contraction, so the result agrees with the axis
    projector to round-off, not bitwise.  ``μ_x`` pieces are zero-padded to
    ``p·⌈μ/P⌉``.
    """
    from common.shard_map import shard_map

    ax_x, ax_y = axes
    nk, nb, ns, rb, mu_l, mu_r, p, _ = _projector_shapes(
        mesh_xy, face_shape, axes, channels, right_face_shape, band_extent,
        row_block, square=True)
    nch = 1 if channels == "none" else 2
    b_loc, mul = nb // p, mu_l // p
    slab_piece = padded_axis(mul, p, name="face slab centroid piece")
    slab_pad, q = slab_piece.pad, slab_piece.carrier // p
    # Admitted against the widest x block's operator tile (the whole tile
    # when the operator comes in one piece).
    w = face_projection_chunk(nk=nk, nb=nb, ns=ns, mu_left=p * rb,
                              mu_right=mu_r, p=p, channels=nch)
    n_chunks = b_loc // w
    both = (ax_x, ax_y)
    transpose = [(i * p + j, j * p + i) for i in range(p) for j in range(p)]
    left_spec, o_spec, right_spec = (
        P(None, ax_x, None, ax_y), P(None, None, ax_x, None, ax_y),
        P(None, None, ax_x, ax_y))
    slab_spec, rt_spec, acc_spec = (
        P(None, None, None, both), P(None, None, ax_y, ax_x), P(None, both))

    def prepare_body(psi_l, psi_r):
        lt = jax.lax.ppermute(psi_l, both, transpose)     # (m_y, s, μ_x)
        rt = jax.lax.ppermute(psi_r, both, transpose)     # (s', ν_y, n_x)
        lt = jnp.pad(lt, ((0, 0), (0, 0), (0, 0), (0, slab_pad)))
        slab = jnp.conj(jax.lax.all_to_all(
            lt, ax_y, split_axis=3, concat_axis=1, tiled=True))
        return slab, rt

    prepare = jax.jit(shard_map(
        prepare_body, mesh=mesh_xy, in_specs=(left_spec, right_spec),
        out_specs=(slab_spec, rt_spec), check_vma=False))

    def accumulate_body(slab, rt, O, acc, *, rows):
        if rows is not None:                 # the block's rows of each slab piece
            slab = slab[..., rows[0]:rows[0] + rows[1]]
        ops = ((O,) if channels == "none" else
               (jnp.real(O).astype(O.dtype), jnp.imag(O).astype(O.dtype)))
        if acc is None:
            acc = jnp.zeros((nch, nk, nb, nb), O.dtype)

        def gather(j):
            cols = jax.lax.dynamic_slice_in_dim(rt, j * w, w, axis=3)
            return jax.lax.all_gather(cols, ax_x, axis=3, tiled=True)

        def chunk_part(r_chunk):
            t = jnp.stack([jnp.einsum("ksmtn,ktnc->ksmc", o, r_chunk)
                           for o in ops])
            if rows is None:                 # x blocks carry their padding rows
                t = jnp.pad(t, ((0, 0),) * 3 + ((0, slab_pad), (0, 0)))
            t = jax.lax.psum_scatter(t, ax_y, scatter_dimension=3,
                                     tiled=True)
            return jnp.einsum("kasm,eksmc->ekac", slab, t)

        if n_chunks == 1:
            return acc + chunk_part(gather(0))
        # Chunk j holds local columns [j·w, (j+1)·w) of every n_x block.
        acc = acc.reshape(nch, nk, nb, p, b_loc)

        def add(acc, part, j):
            part = part.reshape(nch, nk, nb, p, w)
            here = jax.lax.dynamic_slice_in_dim(acc, j * w, w, axis=4)
            return jax.lax.dynamic_update_slice_in_dim(acc, here + part,
                                                       j * w, axis=4)

        def step(carry, j):
            acc, current = carry
            ahead = gather(j + 1)
            return (add(acc, chunk_part(current), j), ahead), None

        (acc, last), _ = jax.lax.scan(step, (acc, gather(0)),
                                      jnp.arange(n_chunks - 1), unroll=1)
        acc = add(acc, chunk_part(last), n_chunks - 1)
        return acc.reshape(nch, nk, nb, nb)

    compiled = {}

    def accumulate(faces, O, acc, *, rows):
        if rows is not None and (rows[2], rows[3]) != (q, p):
            raise ValueError(
                f"face projector: x block {rows} must take slab pieces "
                f"(xs, xn) = ({q}, {p}) (face_row_blocks)")
        key = (rows, acc is None)
        if key not in compiled:
            body = functools.partial(accumulate_body, rows=rows)
            if acc is None:
                fn = shard_map(lambda s, r, o: body(s, r, o, None),
                               mesh=mesh_xy,
                               in_specs=(slab_spec, rt_spec, o_spec),
                               out_specs=acc_spec, check_vma=False)
            else:
                fn = shard_map(body, mesh=mesh_xy,
                               in_specs=(slab_spec, rt_spec, o_spec,
                                         acc_spec),
                               out_specs=acc_spec, check_vma=False)
            compiled[key] = jax.jit(fn)
        args = (*faces, O) if acc is None else (*faces, O, acc)
        return compiled[key](*args)

    warm_mesh_cliques(mesh_xy)
    return BandProjector(
        mesh=mesh_xy, axes=axes, channels=channels,
        expected=((nk, nb, ns, mu_l), (nk, ns, mu_l, ns, mu_r),
                  (nk, ns, mu_r, nb)),
        in_specs=(left_spec, o_spec, right_spec), prepare=prepare,
        accumulate=accumulate,
        finish=_band_block_finish(mesh_xy, axes, p, p, nch),
        row_blocks=lambda n: face_row_blocks(mul, p, n))


def _axis_project_kernel(mesh_xy: Mesh, face_shape, axes, *,
                         channels: str = "none", right_face_shape=None,
                         band_extent=None, row_block=None):
    """The ``layout='axis'`` Σ projector: slab partial, ONE reduction.

    ``axis`` operands carry EVERY band with the centroid axis split over one
    mesh axis (``common.wfn_layout.psi_specs('axis')`` as oriented by
    ``Wavefunctions.projection_faces``)::

        psi_left   (nk, m, s, μ)     P(None, None, None, 'x')
        O          (nk, s, μ, s', ν) P(None, None, 'x', None, 'y')
        psi_right  (nk, s', ν, n)    P(None, None, 'y', None)

    so each rank holds a complete ``(μ_x, ν_y)`` slab of the contraction.
    ``accumulate`` contracts that slab locally,

        T[s,μ,n]  = Σ_{s',ν∈y} O[s,μ,s',ν] ψ_r[s',ν,n]
        part[m,n] = Σ_{s,μ∈x}  conj(ψ_l[m,s,μ]) T[s,μ,n],

    and ``finish`` (:func:`reduce_scatter_to_band_block`) sums the
    ``(nb, nb)`` partial over the mesh into ``P(None, 'x', 'y')`` — one
    collective of ``nb²`` per rank per projection, however many operator
    x blocks.  ``channels='split_reim'`` projects ``Re O`` and ``Im O``
    into the same partial stack.
    """
    from common.shard_map import shard_map

    ax_x, ax_y = axes
    nk, nb, ns, _, mu_l, mu_r, px, py = _projector_shapes(
        mesh_xy, face_shape, axes, channels, right_face_shape, band_extent,
        row_block, square=False)
    mul = mu_l // px
    nch = 1 if channels == "none" else 2
    left_spec, o_spec, right_spec = (
        P(None, None, None, ax_x), P(None, None, ax_x, None, ax_y),
        P(None, None, ax_y, None))
    acc_spec = P(None, (ax_x, ax_y))

    prepare = jax.jit(lambda psi_l, psi_r: (jnp.conj(psi_l), psi_r))

    def accumulate_body(psi_l, psi_r, O, acc, *, rows):
        if rows is not None:                 # ψ_l(m, μ_x): the block's rows
            from ffi.fft import x_block_rows
            idx = x_block_rows(rows)
            keep = jnp.asarray(idx < mul)[None, None, None, :]
            psi_l = jnp.where(keep, jnp.take(
                psi_l, jnp.asarray(np.minimum(idx, mul - 1)), axis=3), 0)
        ops = ((O,) if channels == "none" else
               (jnp.real(O).astype(O.dtype), jnp.imag(O).astype(O.dtype)))
        part = jnp.stack([
            jnp.einsum("kasm,ksmb->kab", psi_l,
                       jnp.einsum("ksmtn,ktnb->ksmb", o, psi_r))
            for o in ops])
        return part if acc is None else acc + part

    compiled = {}

    def accumulate(faces, O, acc, *, rows):
        key = (rows, acc is None)
        if key not in compiled:
            body = functools.partial(accumulate_body, rows=rows)
            if acc is None:
                fn = shard_map(lambda l, r, o: body(l, r, o, None),
                               mesh=mesh_xy,
                               in_specs=(left_spec, right_spec, o_spec),
                               out_specs=acc_spec, check_vma=False)
            else:
                fn = shard_map(body, mesh=mesh_xy,
                               in_specs=(left_spec, right_spec, o_spec,
                                         acc_spec),
                               out_specs=acc_spec, check_vma=False)
            compiled[key] = jax.jit(fn)
        args = (*faces, O) if acc is None else (*faces, O, acc)
        return compiled[key](*args)

    return BandProjector(
        mesh=mesh_xy, axes=axes, channels=channels,
        expected=((nk, nb, ns, mu_l), (nk, ns, mu_l, ns, mu_r),
                  (nk, ns, mu_r, nb)),
        in_specs=(left_spec, o_spec, right_spec), prepare=prepare,
        accumulate=accumulate,
        finish=_band_block_finish(mesh_xy, axes, px, py, nch),
        row_blocks=lambda n: face_row_blocks(mul, py, n))


def contract_bands_block_reshard(
    mesh_xy: Mesh,
    *,
    channels: str = "none",
    extra: str = "none",
    axes: tuple[str, str] = ("x", "y"),
    layout: str = "legacy",
    face_shape=None,
    right_face_shape=None,
    face_band_extent=None,
    row_block=None,
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
        real leading-extra operand if you need both.  Under
        ``layout='face'``/``'axis'`` both channels ride one stream.
    extra
        ``"none"`` | ``"leading"`` | ``"minor"`` — position of the
        caller's stack axis E (see canonical layout).  Both orders are
        first-class so the choice stays a measurement
        (wk_REL/contract_bands_notes.md records the microbench).
        ``"minor"`` always takes the XLA plan (structural: the
        contracted axis is not GEMM-reachable without a full-tile
        transpose copy) — use ``"leading"`` where the GEMM plan matters.
        **Legacy layout only** — refused under ``layout='face'``/``'axis'``.
    axes
        Mesh axis names ``(ax_x, ax_y)``; ax_x shards μ/m, ax_y shards
        ν/n.  Default matches every production mesh.
    layout
        ``"face"``: the band-distributed ``psi_nmu`` ``(nk, m_X, s, μ_Y)``
        and ``psi_mun`` ``(nk, s', ν_X, n_Y)`` (``gw.wavefunction_bundle``),
        every operand and transient at 1/P — the stationary-operator
        stream of :func:`_face_project_kernel`.  Needs a square mesh.
        ``"axis"``: band-complete operands (every band on every rank, the
        centroid axis on one mesh axis) — :func:`_axis_project_kernel`'s
        local slab contraction and one band-block reduce-scatter.
        Both require ``face_shape`` and ``extra="none"`` (a caller with
        several projections calls the kernel once per slice).
        ``"legacy"`` (default): the shard_map + psum_scatter body below,
        BYTE-IDENTICAL to the code this module shipped before the
        face carrier existed.
    face_shape
        ``(nk, nb_full, n_rmu, nspinor)`` — required when ``layout=
        'face'``/``'axis'``.  Fixes the local tile shapes and the face
        stream's band-chunk count at this call.
    right_face_shape
        Optional ``(nk, nb_full, n_rmu_right, nspinor)`` for a rectangular
        operator.  Omit for the historical square projection.  The two
        endpoints must share ``nk``, ``nb_full`` and ``nspinor``; only their
        centroid extents may differ.
    face_band_extent
        The projected band extent (the Σ window's padded carrier), when the
        ψ operands are sliced below ``nb_full``.
    row_block
        Face/axis: the widest x block's local μ rows (``xn·bx`` of
        :func:`face_row_blocks`) a caller projects through
        :class:`BandProjector`'s ``prepare``/``accumulate``/``finish`` — each
        ``O`` block ``(nk, ns, P_x·xn·bx, ns, ν)``, every spin; it sizes the
        face stream's band chunk.  Default: the whole local μ extent.

    Returns
    -------
    ``project(psi_left, O, psi_right)`` — safe to jit / trace into larger
    kernels.  Legacy: a shard_map'd callable, output per the canonical
    layout table.  Face/axis: a :class:`BandProjector`, output
    ``(nk, m, n)`` at ``P(None, ax_x, ax_y)`` — the SAME output spec as
    legacy's.
    """
    if layout not in ("legacy", "face", "axis"):
        raise ValueError(
            f"contract_bands_block_reshard: layout must be 'legacy', "
            f"'face' or 'axis', got {layout!r}")
    if layout in ("face", "axis"):
        if channels not in ("none", "split_reim"):
            raise ValueError(
                f"contract_bands_block_reshard(layout='face'): channels="
                f"{channels!r} not in ('none', 'split_reim')")
        if extra != "none":
            raise NotImplementedError(
                f"contract_bands_block_reshard(layout={layout!r}): extra="
                f"{extra!r} is not ported — the face/axis projectors have "
                "no batched-stack axis; a caller with several projections "
                "to make (band brackets) calls this kernel once per slice "
                "(gw.ppm_tau_kernel._bracketed_face).  channels="
                "'split_reim' is served.")
        if face_shape is None:
            raise ValueError(
                "contract_bands_block_reshard(layout='face') requires "
                "face_shape=(nk, nb_full, n_rmu, nspinor)")
        build = (_face_project_kernel if layout == "face"
                 else _axis_project_kernel)
        return build(mesh_xy, face_shape, axes, channels=channels,
                     right_face_shape=right_face_shape,
                     band_extent=face_band_extent, row_block=row_block)

    from common.shard_map import shard_map

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

    use_ffi = bands_gemm_ffi_enabled()
    if use_ffi:
        # The REQUIRED path (decisions.md 2026-08-01).  Two structural
        # exclusions keep the XLA plan, and only these — neither is a
        # demotion to a duplicate:
        # * non-CPU mesh — the dial does not exist there (host symbol
        #   table only; XLA:GPU's dot lowering already hits cuBLAS, which
        #   is optimal).  Silent BY DESIGN, reason recorded in the gate's
        #   ``silent_platform_demote`` field; ``Gate.resolve`` returns
        #   None for it and announce-or-REFUSES everything else (a
        #   missing handler on a CPU mesh raises — startup enforcement
        #   normally caught it already at initialize_communicator_stack).
        # * extra='minor' — THIS module's fact, about the primitive's
        #   operand layout rather than about BLAS: with the stack axis
        #   minor the contracted (s', nu) axis is not reachable by a
        #   strided batched GEMM without a full-tile transpose copy
        #   (which would cost what the handler saves).  Use
        #   extra='leading' where the GEMM plan matters.
        if _BANDS_GEMM_GATE.resolve(mesh_xy) is None or extra == "minor":
            use_ffi = False

    # -- rank-local GEMM helpers ---------------------------------------
    # The LARGE right contraction, optionally through the FFI handler.
    # Reshapes around the FFI call are index-preserving (contiguous
    # row-major flattening of adjacent axes) — free at the XLA level.

    def _ffi_dtypes_ok(o_l, psi_l):
        """The GEMM handler serves ALL FOUR BLAS precisions since the
        2026-07-29 owner order — f64/f32/c128/c64, dispatched on the
        buffer dtype onto cblas_{d,s,z,c}gemm[_batch].  The BSE fp32-GMRES
        (complex64) path rides the handler.

        A genuinely unserveable dtype (f16, bf16, c256, or a mismatched
        pair that the de-promotion policy should have split upstream)
        REFUSES with the fix named — the required layer never silently
        demotes (decisions.md 2026-08-01); LORRAX_BANDS_GEMM_FFI=0 is the
        announced debug escape onto the XLA lowering."""
        ok = (o_l.dtype == psi_l.dtype
              and o_l.dtype in (jnp.float64, jnp.float32,
                                jnp.complex128, jnp.complex64))
        if ok:
            return ok
        raise TypeError(
            f"LORRAX_BANDS_GEMM_FFI: the vendor-BLAS GEMM host handler "
            f"serves f64/f32/c128/c64, got O {o_l.dtype} × ψ "
            f"{psi_l.dtype}.  A MISMATCHED pair means the de-promotion "
            f"policy did not split a mixed real/complex contraction "
            f"upstream (a bug — report it); a half/extended precision "
            f"operand is genuinely unserved.  LORRAX_BANDS_GEMM_FFI=0 is "
            f"the announced debug escape onto the XLA lowering — the "
            f"required layer never demotes silently.")

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
                    out_specs=out_spec, check_vma=False)

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
        from runtime.padding import authenticate_padded_axis
        authenticate_padded_axis(
            m, m, mesh_xy, name="band projector left carrier",
            spec=P(None, ax_x, None, None), axis=1)
        authenticate_padded_axis(
            n, n, mesh_xy, name="band projector right carrier",
            spec=P(None, None, None, ax_y), axis=3)

    def _project(psi_left, o, psi_right):
        _check_shapes(psi_left, o, psi_right)
        return _sm(psi_left, o, psi_right)

    # Policy 4: create this mesh's MPI cliques on the MAIN thread now, so the
    # pool-worker collectives inside the returned kernel hit XLA's clique
    # cache instead of its MPI_Is_thread_main guard.  No-op off impl=mpi.
    warm_mesh_cliques(mesh_xy)

    return _project
