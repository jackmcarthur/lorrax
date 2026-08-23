"""Canonical wavefunction storage for ISDF-basis GW calculations.

``Wavefunctions`` carries a static ``layout`` tag (pytree aux/meta data —
NOT a traced value) that selects one of two mutually exclusive
representations.  ``low_mem_bands = false`` (the deck default) resolves to
``layout = "legacy"`` on the exact construction path this module has
always used; ``low_mem_bands = true`` resolves to ``layout = "face"``, a
different set of fields entirely.  A jit that closes over a ``Wavefunctions``
compiles a SEPARATE specialization per layout (the tag is meta, so it is
part of the pytree treedef), never a value branch inside one kernel — see
``reports/gwjax_low_mem_bands_audit_2026-08-22/report.md`` §5.

``layout = "legacy"`` — four device-distributed copies of ψ_nk(r_μ), one
for each combination of {device axis} × {memory layout}:

  psi_xn : (nk, s, μ_X, n)  bands fast, μ on X  →  G/χ₀ LHS (conj)
  psi_xr : (nk, n, s, μ_X)  centroids fast, μ on X  →  Σ projection LHS (conj)
  psi_yr : (nk, n, s, μ_Y)  centroids fast, μ on Y  →  G/χ₀ RHS
  psi_yn : (nk, s, μ_Y, n)  bands fast, μ on Y  →  Σ projection RHS

In all four layouts the spinor index s sits adjacent to the centroid
index μ, so contractions that sum over (s, μ) pairs sweep contiguous memory.
All four copies store the *un-conjugated* ψ; consumers that need ψ*
apply :func:`jnp.conj` themselves.  Per-rank residency:
``2·S/Px + 2·S/Py`` where ``S = 16·nk·nspinor·nb·nmu`` (one global
complex128 ψ image) — see :func:`PSI_XN_SPEC` and its three siblings.

``layout = "face"`` — exactly TWO copies, both 2-D sharded on the full
(X, Y) mesh, both un-conjugated:

  psi_nmu : (nk, n, s, μ)  P(None, 'x', None, 'y')  →  the (n, (s,μ)) face
  psi_mun : (nk, s, μ, n)  P(None, None, 'x', 'y')  →  the ((s,μ), n) face

Flattening ``(s, μ)`` only at the GEMM seam gives the two ``P(None,'x','y')``
matrices every ``N,N`` cuBLASMp multiplication needs — see the audit's
verdict for why ONE 2-D-sharded orientation is not enough (transposing a
``P('x','y')`` array changes its physical orientation, and multi-rank
cuBLASMp refuses transpose modes).  Per-rank residency: ``2·S/(Px·Py)`` —
a ``2·√P`` reduction against legacy on a square mesh.  Legacy consumers
that read ``psi_xn``/``psi_xr``/``psi_yr``/``psi_yn`` directly are not
ported here; the four accessor METHODS below (``.xn()``/``.xr()``/
``.yr()``/``.yn()``) refuse by name under ``layout = "face"`` rather than
silently rebuilding a legacy replica.  Raw field access on the four legacy
arrays under ``layout = "face"`` gets ``None`` (an unported consumer's
``AttributeError`` on it is the correctness backstop, not the intended
refusal path — the intended one is the deck-level envelope that keeps such
a consumer from ever being reached under ``low_mem_bands = true``).
"""
from __future__ import annotations

import functools
import math
from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

from runtime.padding import round_up, spec_divisor


# ---------------------------------------------------------------------------
# Band-edge bookkeeping
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class BandSlices:
    """Precomputed local slices from the five canonical band edges.

    **THE single source of truth for band windows.**  ``Meta`` supplies
    the five edges (``meta.band_edges``) and nothing else; the duplicate
    ``Meta.band_ranges`` namespace that used to shadow this — with a
    *different* and unused ``sigma`` convention — was deleted (AD).

    Band edges (global indices):
        b0  lowest band in the calculation
        b1  start of mixed valence/conduction sigma region
        b2  LUMO (first unoccupied)
        b3  end of the sigma/QP evaluation window
        b4  highest band (end of full computational window)

    Two further edges since 2026-08-16, the χ / Σ band-count split:
        b4_chi    top of the χ0/W band sum   (``number_bands_chi``)
        b4_sigma  top of the Σ band sum      (``number_bands_sigma``)

    ``b4`` is ``max(b4_chi, b4_sigma)`` PADDED to the world size — the extent
    the ψ is loaded over and therefore the window the ISDF ζ fit is built
    for.  On an unsplit deck all three coincide and every slice below is
    exactly what it was before the split existed.

    **``full`` IS NOT THE Σ BAND SUM.**  It is the ALLOCATION extent: what
    was loaded, what ζ was fitted on, what the four ψ copies are shaped by.
    Before the split those were the same number as the Σ sum and the name
    got used for both.  The Σ band sum is :attr:`sigma_sum`; the χ0
    conduction leg is :attr:`cond`.  A consumer that reaches for ``full`` to
    mean "the bands my sum runs over" is asking for the larger consumer's
    count and will silently ignore the split.

    All slices are LOCAL (relative to b0).
    """
    b0: int
    b1: int
    b2: int
    b3: int
    b4: int
    val:   slice   # [0, b2-b0)          valence
    cond:  slice   # [b2-b0, b4_chi-b0)  chi0 conduction leg
    sigma: slice   # [0, b3-b0)          QP evaluation window
    full:  slice   # [0, b4-b0)          everything LOADED (== the ISDF window)
    occ:   slice   # [0, b2-b0)          occupied
    b4_chi: int = 0        # 0 == "not split": resolves to b4
    b4_sigma: int = 0
    sigma_sum: slice = slice(0, 0)   # [0, b4_sigma-b0)  the Sigma band sum
    #: [b2-b0, b4-b0) — conduction bands over the LOADED window, i.e. the
    #: UNION of what χ0 and Σ reach.  Exactly ``cond`` on an unsplit deck.
    #: Its one consumer is the minimax τ-axis (``build_static_quadrature``),
    #: which certifies a 1/x rule on ``[x_min, x_max]`` and is then reused by
    #: BOTH stages: an interval built from the χ conduction top alone would
    #: not cover Σ's transitions on a deck whose Σ count is the larger one,
    #: and the resulting quadrature error is invisible at the seam that
    #: caused it.
    cond_all: slice = slice(0, 0)

    @classmethod
    def from_band_edges(cls, b0: int, b1: int, b2: int, b3: int, b4: int,
                        *, b4_chi: int | None = None,
                        b4_sigma: int | None = None) -> BandSlices:
        if not (b0 <= b1 <= b2 <= b3 <= b4):
            raise ValueError(f"Invalid band edges: {(b0, b1, b2, b3, b4)}")
        nb_full = b4 - b0
        b4_chi = b4 if b4_chi in (None, 0) else int(b4_chi)
        b4_sigma = b4 if b4_sigma in (None, 0) else int(b4_sigma)
        for name, edge in (("b4_chi", b4_chi), ("b4_sigma", b4_sigma)):
            if not (b2 <= edge <= b4):
                raise ValueError(
                    f"Invalid {name}={edge}: it is a band-sum top inside the "
                    f"loaded window and must satisfy b2={b2} <= {name} <= "
                    f"b4={b4}.  (Below b2 the sum would not reach the first "
                    f"unoccupied band; above b4 there is no loaded ψ.)")
        if max(b4_chi, b4_sigma) != b4:
            raise ValueError(
                f"Invalid split: max(b4_chi={b4_chi}, b4_sigma={b4_sigma}) = "
                f"{max(b4_chi, b4_sigma)} != b4={b4}.  b4 is the PADDED top "
                f"of the larger consumer, so the larger consumer must own it "
                f"exactly; see common.meta.Meta.from_system.")
        return cls(
            b0=b0, b1=b1, b2=b2, b3=b3, b4=b4,
            val=slice(0, b2 - b0),
            cond=slice(b2 - b0, b4_chi - b0),
            sigma=slice(0, b3 - b0),
            full=slice(0, nb_full),
            occ=slice(0, b2 - b0),
            b4_chi=b4_chi,
            b4_sigma=b4_sigma,
            sigma_sum=slice(0, b4_sigma - b0),
            cond_all=slice(b2 - b0, nb_full),
        )

    @property
    def nb_full(self) -> int:
        """Bands LOADED (== the ISDF ζ-fit window).  Not the Σ sum."""
        return self.b4 - self.b0

    @property
    def nb_sigma(self) -> int:
        """Bands in the QP EVALUATION window (b3-b0).  Not the Σ band sum."""
        return self.b3 - self.b0

    @property
    def nb_chi(self) -> int:
        """Bands in the χ0/W band sum."""
        return self.b4_chi - self.b0

    @property
    def nb_sigma_sum(self) -> int:
        """Bands in the Σ band sum (the count the extrapolation brackets)."""
        return self.b4_sigma - self.b0

    @property
    def is_split(self) -> bool:
        """True when χ and Σ were given different band counts."""
        return self.b4_chi != self.b4_sigma

    @property
    def sigma_range(self) -> tuple[int, int]:
        """Global (start, end) for sigma band window: (b0, b3)."""
        return (self.b0, self.b3)

    @property
    def full_range(self) -> tuple[int, int]:
        """Global (start, end) for the LOADED band window: (b0, b4)."""
        return (self.b0, self.b4)


# ---------------------------------------------------------------------------
# Sharding specs for the four ψ copies (2-D mesh with axes 'x', 'y').
# ---------------------------------------------------------------------------
PSI_XN_SPEC = P(None, None, 'x', None)   # (nk, s, μ_X, n)
PSI_XR_SPEC = P(None, None, None, 'x')   # (nk, n, s, μ_X)
PSI_YR_SPEC = P(None, None, None, 'y')   # (nk, n, s, μ_Y)
PSI_YN_SPEC = P(None, None, 'y', None)   # (nk, s, μ_Y, n)

# ---------------------------------------------------------------------------
# Sharding specs for the two FACE copies (``low_mem_bands = true``).  Both
# 2-D sharded on the full (x, y) mesh; both un-conjugated.  See the module
# docstring and reports/gwjax_low_mem_bands_audit_2026-08-22/report.md for
# why these two orientations (not one, not a transpose of one another) are
# the portable representation under a cuBLASMp N,N-only multi-rank GEMM.
PSI_NMU_SPEC = P(None, 'x', None, 'y')   # (nk, n_X, s, μ_Y)
PSI_MUN_SPEC = P(None, None, 'x', 'y')   # (nk, s, μ_X, n_Y)

# ---------------------------------------------------------------------------
# Sharding specs for the intermediate tensors that flow between chi0 / W /
# sigma / cohsex kernels.  Kept here (not in each consumer) so every
# module sees the same canonical layout and a reshard-mismatch is caught
# at import time rather than at HLO compile.
# ---------------------------------------------------------------------------
# G(k) in 7-D FFT-box form, used by chi0's build_G and sigma's Gij
# construction.  (nkx, nky, nkz, s, μ_X, spinor, μ_Y) with μ on the
# (x, y) mesh.  The s-axis and spinor-axis are replicated because they
# sum contiguously in downstream einsums.
G_FFT7D_SPEC = P(None, None, None, None, 'x', None, 'y')

# V_q / W_q in 5-D k-space form: (nkx, nky, nkz, μ_X, μ_Y) both μ-axes
# sharded.  Used by compute_vcoul, w_isdf's V_q factory, and the W_q
# input to sigma.
V_FFT5D_SPEC = P(None, None, None, 'x', 'y')

# chi(q) / σ^τ(k, μ, ν) in 5-D flat-q or k-sharded form: both μ axes on
# the (x, y) mesh, q/k replicated.  Used by the inner tau kernel and
# the chi0 → W solve.
CHI_Q_SPEC = P(None, None, None, 'x', 'y')

# G(k) in 5-D flat-k form — the output of make_flat_k_fftn when the
# upstream op operates on a 3-D k-grid view.  (nk_flat, s, μ_X, spinor,
# μ_Y).
G_FLATK_SPEC = P(None, None, 'x', None, 'y')

# chi / W in R-space, flat-k form, used by the chi0-to-W pipeline post
# iFFT: (nk_flat, μ_X, μ_Y).
CHI_R_SPEC = P(None, 'x', 'y')


# ---------------------------------------------------------------------------
# Wavefunction storage
# ---------------------------------------------------------------------------

#: Valid ``Wavefunctions.layout`` tags.  Anything else refuses at
#: construction (:meth:`Wavefunctions.__post_init__`).
_LAYOUTS = ('legacy', 'face')


@dataclass
class Wavefunctions:
    """ψ_nk(r_μ) spanning [b0, b4), in ONE of two mutually exclusive
    representations selected by the static ``layout`` tag.  See the module
    docstring for the full field/spec table of each.

    ``layout = "legacy"`` (default — ``low_mem_bands = false``) populates
    ``psi_xn``/``psi_xr``/``psi_yr``/``psi_yn``; ``psi_nmu``/``psi_mun``
    are ``None``.  ``layout = "face"`` (``low_mem_bands = true``) is the
    reverse.  Construct through :func:`build_wavefunctions` (legacy) or
    :func:`build_wavefunctions_face` (face) rather than this dataclass
    directly — both name every field explicitly, including the ``None``
    half, so a caller cannot omit the layout tag's inverse by accident.
    """

    enk: jax.Array       # (nk, nb_full) replicated
    #: (nk, nb_full) float64 replicated.  Storage is a WEIGHT, but only ρ
    #: (``qsgw_density.rho_from_wfns``) consumes it as one today.  χ₀ takes
    #: its val/cond split from a BAND-INDEX CUT (``minimax_screening.py``
    #: :943-944, ``s.val``/``s.cond``) and the one Σ site that does read
    #: this array thresholds it (``ppm_sigma.py``:181, ``occ_full > 0.5``).
    #: So a fractional-occupation port must change those as well: filling
    #: ``occ`` alone leaves χ₀ and Σ on a step function while ρ alone is
    #: smeared, and nothing would flag the disagreement.  **Static Σ_x/Σ_SX
    #: and V_H are DONE (2026-08-15):** they read the carried
    #: ``OccupationState`` through ``cohsex_sigma.build_Gij``'s ``diag(f)``
    #: branch, not this array, and fall back to the integer projector
    #: bit-for-bit when no state is carried.
    occ: jax.Array
    slices: BandSlices
    # ---- legacy (four single-axis copies); None under layout="face" ----
    psi_xn: jax.Array | None = None   # (nk, s, μ_X, n)
    psi_xr: jax.Array | None = None   # (nk, n, s, μ_X)
    psi_yr: jax.Array | None = None   # (nk, n, s, μ_Y)
    psi_yn: jax.Array | None = None   # (nk, s, μ_Y, n)
    # ---- face (two 2-D-sharded copies); None under layout="legacy" -----
    psi_nmu: jax.Array | None = None  # (nk, n_X, s, μ_Y)
    psi_mun: jax.Array | None = None  # (nk, s, μ_X, n_Y)
    #: STATIC (pytree meta, never traced).  "legacy" | "face".
    layout: str = "legacy"

    def __post_init__(self) -> None:
        if self.layout not in _LAYOUTS:
            raise ValueError(
                f"Wavefunctions: layout={self.layout!r} not in {_LAYOUTS}.")

    def _require_legacy(self, accessor: str) -> None:
        """Refuse BY NAME rather than silently rebuild a legacy replica.

        Called at the top of each ``.xn()``/``.xr()``/``.yr()``/``.yn()``
        accessor body.  ``self.layout`` is pytree META (a Python string,
        never a traced value) so this branch resolves at jax TRACE time:
        under ``layout="legacy"`` it costs nothing (dead code eliminated
        before any op is emitted, so the legacy jaxpr this accessor
        produces is exactly what it always was); under ``layout="face"``
        it raises before any op is emitted, which surfaces to the caller
        as a normal Python exception out of the ``jax.jit`` call.
        """
        if self.layout != 'legacy':
            raise ValueError(
                f"Wavefunctions.{accessor}() is a legacy-layout accessor "
                f"and refuses under layout={self.layout!r}: this bundle "
                f"does not store psi_{accessor}.  Face layout stores "
                f"psi_nmu/psi_mun (both 2-D sharded, P(None,'x','y') at "
                f"the (s,μ) GEMM seam) — read those directly, or route "
                f"through the layout-dispatching G/projection kernel "
                f"rather than reconstructing a legacy view.")

    # Slice accessors — bands is a Python ``slice`` (hashable in 3.12+) so
    # jit can take it as static_argname.  Without these jits each accessor
    # call (used heavily by chi/W/Σ) emits a fresh eager-pjit ``gather``,
    # producing a tail of cache misses (~17/run on Si 4×4×4).
    @functools.partial(jax.jit, static_argnames=('bands',))
    def xn(self, bands: slice) -> jax.Array:
        self._require_legacy('xn')
        return self.psi_xn[:, :, :, bands]

    @functools.partial(jax.jit, static_argnames=('bands',))
    def xr(self, bands: slice) -> jax.Array:
        self._require_legacy('xr')
        return self.psi_xr[:, bands, :, :]

    @functools.partial(jax.jit, static_argnames=('bands',))
    def yr(self, bands: slice) -> jax.Array:
        self._require_legacy('yr')
        return self.psi_yr[:, bands, :, :]

    @functools.partial(jax.jit, static_argnames=('bands',))
    def yn(self, bands: slice) -> jax.Array:
        self._require_legacy('yn')
        return self.psi_yn[:, :, :, bands]

    @functools.partial(jax.jit, static_argnames=('bands',))
    def band_mask(self, bands: slice) -> jax.Array:
        """Boolean ``(nk, nb_full)`` band-identity mask, True inside
        ``bands`` — the face-layout bring-up path for band windows
        (report §3, obstacle 3).  ``psi_nmu``/``psi_mun`` span the FULL
        loaded [b0,b4) range and cannot legally be sliced to an arbitrary
        logical window (the band axis is mesh-sharded on one face or the
        other, and a window need not be mesh-divisible).  A window is
        instead expressed as a WEIGHT that is exactly zero outside it —
        this mask is that weight's boolean form, meant for
        ``greens_function_kernel.build_G_tau``'s ``mask=`` under
        ``layout='face'`` (the SAME parameter legacy Σ already uses for
        its own band-identity selector, ``mask_A``).

        COST, named rather than hidden: every masked face call pays the
        FULL nb_full GEMM, not the windowed one — the bring-up path
        "repeats full-band GEMM work for val/cond and for every Sigma
        bracket" (report §3's own words).  A future canonical distributed
        ``pack_band_window`` would remove this; it does not exist yet.

        Legal under EITHER layout (only ``self.enk``'s shape is read),
        but only a face caller needs it — a legacy accessor already
        returns the exact physical slice and has no residual bands to
        zero.  Zero-weighted bands are NOT automatically safe for an
        eigensolve or an occupation sum fed by this mask's caller; this
        helper only guarantees the G/Σ *contraction* ignores them.
        """
        nb_full = int(self.enk.shape[1])
        lo = int(bands.start or 0)
        hi = int(bands.stop if bands.stop is not None else nb_full)
        idx = jnp.arange(nb_full)
        row = (idx >= lo) & (idx < hi)
        return jnp.broadcast_to(row[None, :], self.enk.shape)


# Register as JAX pytree so Wavefunctions can be passed to @jax.jit functions.
# Array fields are traced; slices/layout (static metadata) are compile-time
# constants — ``layout`` as a META field (not data) is what makes it
# SPECIALIZE a jit into a separate compiled kernel per layout rather than
# becoming a traced value a kernel branches on (report §5).  A ``None``
# array field (the unused half of the legacy/face split) is a valid, empty
# pytree leaf-set; it costs nothing and traces to nothing.
jax.tree_util.register_dataclass(
    Wavefunctions,
    data_fields=['psi_xn', 'psi_xr', 'psi_yr', 'psi_yn',
                 'psi_nmu', 'psi_mun', 'enk', 'occ'],
    meta_fields=['slices', 'layout'],
)


def face_kernel_kwargs(wfns: "Wavefunctions") -> dict:
    """``{}`` under ``layout='legacy'``; ``{"layout": "face", "face_shape":
    (nk, nb_full, n_rmu, nspinor)}`` under ``layout='face'``.

    THE single source of truth for the layout/face_shape kwargs every
    layout-dispatching kernel factory in this codebase needs
    (``common.contract_bands.contract_bands_block_reshard``,
    ``gw.cohsex_sigma._make_cohsex_kernels``, ``gw.ppm_tau_kernel``'s
    kij/spatial/tau kernel factories, ``gw.mpa.sigma``'s shared tau
    kernel call) — read off ``wfns.psi_mun``'s own shape rather than
    threaded in by every call site, since the bundle already carries it.
    Originally ``cohsex_sigma._face_kwargs`` (2026-08-22 COHSEX face
    port); moved here and generalised the same day so the dynamic PPM/MPA
    Σ_c(τ) port could reuse it rather than re-deriving the same 4-tuple
    a second time (TASTE microservice rule: a routine used in 2+ places
    gets one owner).
    """
    if wfns.layout != "face":
        return {}
    nk, s, mu, n = wfns.psi_mun.shape
    return {"layout": "face", "face_shape": (nk, wfns.slices.nb_full, mu, s)}


# ---------------------------------------------------------------------------
# Bispinor Σ^B: γ̃ vertex insertion — representation-aware, bundle-owned
# (census row "Bispinor transverse exchange",
# reports/gwjax_low_mem_bands_audit_2026-08-22/report.md).  Both
# consumers below used to live in gw.sigma_x_bispinor, hard-coded to the
# legacy field names; moved here so a caller gets the right field for
# WHICHEVER layout its bundle carries, without re-deriving the mapping.
# ---------------------------------------------------------------------------

#: ``layout -> (direct_field, direct_spin_axis, conj_field, conj_spin_axis)``
#: — the two operands ``greens_function_kernel.build_G`` consumes for that
#: layout (module docstring: "left is the μ/direct side, right is
#: conjugated internally").  The single source of truth for which FIELD
#: plays which G-vertex role, so :func:`with_lorentz_vertices` needs no
#: per-layout branch of its own physics — only this table differs between
#: layouts, and both entries place the spinor axis at the position their
#: own field's shape puts it (see the two ``PSI_*_SPEC`` blocks above).
_G_VERTEX_FIELDS: dict[str, tuple[str, int, str, int]] = {
    "legacy": ("psi_xn", 1, "psi_yr", 2),
    "face":   ("psi_mun", 1, "psi_nmu", 2),
}


def with_lorentz_vertices(
    wfns: "Wavefunctions", mu_L: int, nu_L: int,
) -> "Wavefunctions":
    """γ̃^{mu_L} folded into the G-build's LEFT (direct-μ) vertex operand,
    γ̃^{nu_L} into its RIGHT (conjugated-internally) operand — the
    bispinor Σ^B transverse-exchange trick (``gw.sigma_x_bispinor``'s
    module docstring has the physics derivation; γ̃^0 = I_4 is the charge
    channel, ``mu_L = nu_L = 0`` reproduces today's scalar Σ_X path
    byte-identically — see below).

    REPRESENTATION-AWARE: which FIELD carries the G-build's direct/
    conjugated role is a LAYOUT fact, read from :data:`_G_VERTEX_FIELDS`
    rather than hard-coded per caller.  ``layout='legacy'`` folds into
    ``psi_xn``/``psi_yr`` (:func:`greens_function_kernel._legacy_build_G`'s
    two operands); ``layout='face'`` folds into ``psi_mun``/``psi_nmu``
    (:func:`greens_function_kernel._face_build_G`'s — the SAME argument
    slots, direct-then-conjugated, under the two-face carrier's own field
    names).  This is what lets ``gw.sigma_x_bispinor.compute_sigma_x_bispinor``
    call this ONE function regardless of which layout its
    ``wfns_transverse`` bundle carries, rather than special-casing face
    the way the pre-2026-08-23 version hard-coded legacy.

    Only the two G-vertex operands move.  The PROJECTION operand pair
    (``psi_xr``/``psi_yn`` under legacy, ``psi_nmu``/``psi_mun`` again
    under face — see :func:`project`'s own argument-slot docstring)
    passes through UNCHANGED: the γ̃ vertex sits on the G-build side of
    the kernel chain, not the projection side (physics: ``gw.
    sigma_x_bispinor``'s module docstring).  A caller that projects
    through this SAME bundle (as ``sigma_sx`` does) therefore inserts the
    vertex exactly once, at the G-build seam, never at both seams.

    ``mu_L == 0`` / ``nu_L == 0`` SKIP that field's ``gamma_apply`` call
    entirely rather than applying an identity permutation — so
    ``with_lorentz_vertices(wfns, 0, 0)`` (the CC/charge-channel call)
    returns ``wfns`` itself, no allocation, no dataclass clone.
    """
    if wfns.layout not in _G_VERTEX_FIELDS:
        raise ValueError(
            f"with_lorentz_vertices: unknown layout {wfns.layout!r}, "
            f"expected one of {tuple(_G_VERTEX_FIELDS)}")
    direct_field, direct_axis, conj_field, conj_axis = _G_VERTEX_FIELDS[wfns.layout]

    from common.gamma_matrices import gamma_perm_phase, gamma_apply

    updates: dict = {}
    if int(mu_L) != 0:
        perm, phase = gamma_perm_phase(int(mu_L))
        updates[direct_field] = gamma_apply(
            getattr(wfns, direct_field), perm, phase, axis=direct_axis)
    if int(nu_L) != 0:
        perm, phase = gamma_perm_phase(int(nu_L))
        updates[conj_field] = gamma_apply(
            getattr(wfns, conj_field), perm, phase, axis=conj_axis)
    if not updates:
        return wfns
    import dataclasses
    return dataclasses.replace(wfns, **updates)


def psi_field_names(layout: str) -> tuple[str, ...]:
    """The populated ψ field names for ``layout`` — ``('psi_xn', 'psi_xr',
    'psi_yr', 'psi_yn')`` for ``'legacy'``, ``('psi_nmu', 'psi_mun')`` for
    ``'face'``.  One place naming which fields are live per layout, so
    :func:`bundle_bytes_per_rank` and any future per-field consumer read
    the same list rather than re-deriving it."""
    if layout == "legacy":
        return ("psi_xn", "psi_xr", "psi_yr", "psi_yn")
    if layout == "face":
        return ("psi_nmu", "psi_mun")
    raise ValueError(f"psi_field_names: unknown layout {layout!r}")


def bundle_bytes_per_rank(wfns: "Wavefunctions") -> dict:
    """MEASURED (never modeled) per-rank byte residency of this bundle's
    ψ fields, read off each array's OWN ``.addressable_shards`` — the
    same instrument ``docs/architecture/zeta_fit_face_psi_cct.md``'s own
    verification section prefers over a sharding-blind global-shape read
    (``KNOWN_LORRAX_ISSUES.md``'s ``mem_probe`` row: a global shape
    cannot tell a genuinely 2-D-sharded array from a same-shape
    single-axis one).

    Returns ``{field: bytes, ..., 'total': bytes}`` over
    :func:`psi_field_names` for this bundle's own ``layout``.

    Named consumer: the bispinor Σ^B transverse-centroid bundle DOUBLES
    whichever psi inventory a run already carries — a second, independent
    :class:`Wavefunctions` on a different (usually smaller) centroid set.
    This module's own docstring states the closed form
    (``2·S/Px + 2·S/Py`` legacy, ``2·S/(Px·Py)`` face) but
    ``gw.gflat_memory_model`` never prices it — that planner's own scope
    note is Stages A-F, the ISDF fit through the V_q tensor write, and
    explicitly NOT the post-fit Σ-stage bundle a second call to
    :func:`build_wavefunction_bundle`/:func:`build_wavefunctions_face`
    produces.  Measuring the REAL bundle here, after construction, is
    simpler and strictly more trustworthy than adding a second, parallel
    model of the same number; see ``gw.sigma_x_bispinor``'s disclosure
    print for the call site.
    """
    out: dict = {}
    for f in psi_field_names(wfns.layout):
        arr = getattr(wfns, f)
        if arr is None:
            continue
        out[f] = int(sum(int(s.data.nbytes) for s in arr.addressable_shards))
    out["total"] = int(sum(out.values()))
    return out


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------

def _build_occ(enk_full, slices, efermi):
    """Build occupation array (nk, nb_full), float64 in {0.0, 1.0}.

    ─ NOTE TO FUTURE EDITORS — THE numpy USAGE BELOW IS INTENTIONAL ─
    ``enk_full`` is tiny (nk × nb, usually < 10K doubles).  The original
    all-``jnp`` version did ``jnp.zeros_like(sharded_input) .at[].set``,
    which not only emitted 8 standalone pjits at trace time but also
    had a **non-trivial runtime cost** tied to the cross-device
    scatter — the ``wavefunction_setup`` timing section dropped
    1.79 s → 0.18 s when this function was converted (commit 7781b80,
    2026-04-18).  The D2H on ``enk_full`` is sub-ms, numpy arithmetic
    is immediate, and the ``jnp.asarray`` at the end is a cheap H2D of
    a small replicated array.  DO NOT "fix" back to ``jnp``.
    """
    enk_host = np.asarray(enk_full)
    if efermi is None:
        occ = np.zeros_like(enk_host, dtype=np.float64)
        occ[:, slices.occ] = 1.0
    else:
        occ = (enk_host <= float(efermi)).astype(np.float64)
    return jnp.asarray(occ)


def build_wavefunctions(
    psi_rmu_Y, psi_rmuT_X, *, enk_full, slices, mesh_xy, efermi=None,
) -> Wavefunctions:
    """Assemble the four-copy ``Wavefunctions`` bundle from the two
    centroid-sampled arrays produced by ``load_centroids_band_chunked``.

    Both inputs cover the full band range [b0, b4).

    ``psi_rmu_Y``   (nk, nb, ns, n_rmu)   P(None, None, None, 'y')
                    un-conjugated ψ.
    ``psi_rmuT_X``  (nk, n_rmu, nb, ns)   P(None, 'x', None, None)
                    conjugated ψ* (layout picked by the ISDF pair-density
                    kernel); a single :func:`jnp.conj` here undoes that
                    convention to match the bundle-wide un-conjugated
                    storage.

    No cross-device reshards are emitted: the y-sharded copies are
    derived from ``psi_rmu_Y`` and the x-sharded copies from
    ``psi_rmuT_X``.  Each transpose preserves the μ axis's sharding.
    """
    with mesh_xy:
        # y-sharded copies — both from psi_rmu_Y.
        psi_yr = jax.lax.with_sharding_constraint(
            psi_rmu_Y, NamedSharding(mesh_xy, PSI_YR_SPEC))
        psi_yn = jax.lax.with_sharding_constraint(
            psi_rmu_Y.transpose(0, 2, 3, 1),
            NamedSharding(mesh_xy, PSI_YN_SPEC))

        # x-sharded copies — conj to undo the pair-density kernel's ψ*.
        psi_X = jnp.conj(psi_rmuT_X)
        psi_xn = jax.lax.with_sharding_constraint(
            psi_X.transpose(0, 3, 1, 2),    # (nk, ns, μ_X, nb)
            NamedSharding(mesh_xy, PSI_XN_SPEC))
        psi_xr = jax.lax.with_sharding_constraint(
            psi_X.transpose(0, 2, 3, 1),    # (nk, nb, ns, μ_X)
            NamedSharding(mesh_xy, PSI_XR_SPEC))

        occ_full = _build_occ(enk_full, slices, efermi)
        rep2 = NamedSharding(mesh_xy, P(None, None))
        enk_full = jax.lax.with_sharding_constraint(enk_full, rep2)
        occ_full = jax.lax.with_sharding_constraint(occ_full, rep2)

    return Wavefunctions(
        psi_xn=psi_xn, psi_xr=psi_xr, psi_yr=psi_yr, psi_yn=psi_yn,
        enk=enk_full, occ=occ_full, slices=slices,
    )


def build_wavefunctions_face(
    psi_rmu_Y, psi_rmuT_X, *, enk_full, slices, mesh_xy, efermi=None,
) -> Wavefunctions:
    """Assemble the ``layout="face"`` ``Wavefunctions`` bundle
    (``low_mem_bands = true``) from the SAME two centroid-sampled arrays
    :func:`build_wavefunctions` consumes — ``psi_rmu_Y``/``psi_rmuT_X`` as
    produced by ``load_centroids_band_chunked`` / the ISDF fit's pair-density
    kernel.  See that function's docstring for their shapes/specs.

    Both faces are FREE resharding constraints, no transpose collective and
    no gather, for the same reason the four legacy copies are free (see
    :func:`build_wavefunctions`'s docstring): the band axis n is fully
    REPLICATED in both inputs, so placing it on a mesh axis the array does
    not already use is a local per-rank slice, not communication.

      psi_nmu = psi_rmu_Y, band axis n moved onto 'x'.  ``psi_rmu_Y``'s own
                axis order (nk, n, s, μ) already equals ``PSI_NMU_SPEC``'s
                — no transpose, just the constraint.
      psi_mun = conj(psi_rmuT_X), transposed (nk,μ,n,s) -> (nk,s,μ,n) to
                match ``PSI_MUN_SPEC``'s axis order, band axis n moved
                onto 'y'.

    Per-rank residency after this call is ``2·S/(Px·Py)`` against
    :func:`build_wavefunctions`'s ``2·S/Px + 2·S/Py`` — the memory result
    ``low_mem_bands`` exists for.  Callers on the fresh-fit path should
    drop their reference to ``psi_rmu_Y``/``psi_rmuT_X`` immediately after
    this returns (``del``) so the single-axis fit copies do not sit
    resident alongside the face copies through V/W/Σ.
    """
    with mesh_xy:
        psi_nmu = jax.lax.with_sharding_constraint(
            psi_rmu_Y, NamedSharding(mesh_xy, PSI_NMU_SPEC))
        psi_mun = jax.lax.with_sharding_constraint(
            jnp.conj(psi_rmuT_X).transpose(0, 3, 1, 2),  # (nk, s, μ_X, n)
            NamedSharding(mesh_xy, PSI_MUN_SPEC))

        occ_full = _build_occ(enk_full, slices, efermi)
        rep2 = NamedSharding(mesh_xy, P(None, None))
        enk_full = jax.lax.with_sharding_constraint(enk_full, rep2)
        occ_full = jax.lax.with_sharding_constraint(occ_full, rep2)

    return Wavefunctions(
        psi_nmu=psi_nmu, psi_mun=psi_mun,
        enk=enk_full, occ=occ_full, slices=slices, layout="face",
    )


def wavefunctions_face_from_restart(
    psi_nmu, psi_mun, *, enk_full, slices, mesh_xy, efermi=None,
) -> Wavefunctions:
    """Assemble the ``layout="face"`` bundle from arrays ALREADY read at
    their face specs (``file_io.load_restart_state_from_h5``,
    ``low_mem_bands=True``) — no resharding constraint applied to either
    face, since each was read as its own direct SlabIO hyperslab.  Mirrors
    :func:`build_wavefunctions_face`'s ``occ``/``enk`` handling exactly
    (same ``_build_occ`` call, same replicated placement) so a restart
    bundle and a fresh-fit bundle agree on occupation for the same
    ``efermi``.
    """
    with mesh_xy:
        occ_full = _build_occ(enk_full, slices, efermi)
        rep2 = NamedSharding(mesh_xy, P(None, None))
        enk_full = jax.lax.with_sharding_constraint(enk_full, rep2)
        occ_full = jax.lax.with_sharding_constraint(occ_full, rep2)

    return Wavefunctions(
        psi_nmu=psi_nmu, psi_mun=psi_mun,
        enk=enk_full, occ=occ_full, slices=slices, layout="face",
    )


# ---------------------------------------------------------------------------
# Self-consistent QSGW: rotate the bundle into a new band basis
# ---------------------------------------------------------------------------

#: ``(ψ field, its spec, the array axis carrying the band index)`` for the
#: four copies, in the order :class:`Wavefunctions` stores them.  The band
#: axis is not restated as a literal anywhere below; it is read from here.
_PSI_LAYOUTS = (
    ('psi_xn', PSI_XN_SPEC, 3),
    ('psi_xr', PSI_XR_SPEC, 1),
    ('psi_yr', PSI_YR_SPEC, 1),
    ('psi_yn', PSI_YN_SPEC, 3),
)


def band_mix_spec(contract_axis: str) -> P:
    """``U`` layout for the ψ rotation: ``(nk, nb, nb)`` with the
    CONTRACTED index m on ``contract_axis`` and n replicated.

    NOT ``gw.qsgw_density.band_rotation_spec`` (``P(None,'x','y')``), and
    the difference is forced rather than chosen.  That spec is right for
    the G-sphere copy ``(nk, nb, ns, ngkmax)``, whose ONLY sharded axis is
    the band axis, so m and n can take one mesh axis each.  Here μ already
    owns a mesh axis in all four copies (``PSI_*_SPEC`` above), so n has
    nowhere to go: n on the free mesh axis would collide with μ in the
    output and the collision can only be paid for by resharding ψ — the
    largest object in the bundle, and the one thing this layout exists to
    leave alone.  m on the axis μ does NOT own contracts each rank's own
    band slice against its existing μ slice, reduces along that one mesh
    axis, and lands the result in the INPUT layout.

    The price of a replicated n, stated because it is real: per-rank U is
    ``nb·(nb/p)`` instead of the ``(nb/px)·(nb/py)`` that
    ``band_rotation_spec`` gets.  Per k at nb=640 on an 8×8 mesh that is
    819 kB against 102 kB — and against 6.5 MB replicated, which is the
    comparison that matters, since the replicated form is the
    ``(nk, nb, nb)`` object that reaches 9.2 GB at nb=2000/nk=144.
    """
    return P(None, contract_axis, None)


def _spec_entries(spec: P, ndim: int) -> list:
    """``spec`` as a length-``ndim`` list, so it can be indexed by ARRAY axis.

    JAX can hand back a PartitionSpec with trailing ``None`` entries trimmed
    — the hazard ``common.wfn_transforms._sharding_key`` normalises against.
    Here it would ``IndexError`` on the band axis of the two band-last
    copies rather than merely recompile.
    """
    entries = list(tuple(spec))
    if len(entries) > ndim:
        raise ValueError(f"spec {spec} is longer than ndim={ndim}")
    return entries + [None] * (ndim - len(entries))


def _contract_axis(mesh: Mesh, psi_spec: P) -> str:
    """The mesh axis U's contracted index m must occupy for ``psi_spec``.

    Whichever axis of the 2-D mesh that copy's μ index does not already
    own — DERIVED from the spec and the mesh, not passed in.  There is
    exactly one correct answer per copy, so an argument would be a knob
    with a single right setting and nothing to check it against; and it
    would have to be got right four times per call.
    """
    named = [e for e in tuple(psi_spec) if e is not None]
    if len(named) != 1 or not isinstance(named[0], str):
        raise ValueError(
            f"rotate_wavefunctions: ψ spec {psi_spec} must place exactly one "
            f"mesh axis (μ's); got {named}.")
    free = [a for a in mesh.axis_names if a != named[0]]
    if len(free) != 1:
        raise ValueError(
            f"rotate_wavefunctions needs a 2-D mesh with one axis free of μ; "
            f"mesh axes {tuple(mesh.axis_names)} minus μ's '{named[0]}' left "
            f"{free}.")
    return free[0]


def _place_U(U, mesh: Mesh, nb_pad: int):
    """``U`` onto the mesh without ever building a replicated ``(nk,nb,nb)``.

    A ``jax.Array`` is handed through untouched: the kernel constrains it
    to :func:`band_mix_spec` per copy and XLA emits the minimal reshard
    from whatever layout it arrived in — a free local slice from a
    replicated input, a small all-gather from ``band_rotation_spec``.
    Resharding it here to a canonical layout first would ADD a collective
    on the replicated-input path (local slice, then gather back what the
    rank already had).

    A HOST array — what ``sc_iteration``'s k-star broadcast produces on a
    reduced k-set — is placed straight at
    ``gw.qsgw_density.band_rotation_spec``, reused rather than re-spelled:
    it is the smallest of the three layouts (nb²/(px·py) per rank) and the
    layout the distributed eigh already emits.  ``jnp.asarray`` here would
    build the whole ``(nk, nb, nb)`` on ONE device first — 9.2 GB at
    nb=2000/nk=144, the object this path exists to avoid — and at P>1 it
    would be a single-device array, which is an operand-sharding error
    against the mesh-sharded ψ rather than a slow success.
    """
    if isinstance(U, jax.Array):
        return U
    from common.collectives import device_put_process_local
    from gw.qsgw_density import band_rotation_spec

    U_np = np.asarray(U, dtype=np.complex128)
    n_in = int(U_np.shape[1])
    if n_in != nb_pad:
        # Both axes, on the host: ``band_rotation_spec`` shards m AND n, so
        # both must divide the mesh before the placement.  Zero pad — a pad
        # ROW would rotate physical bands into a pad band and a pad COLUMN
        # would build a pad QP state out of physical ones; zeros keep both
        # exactly zero, and the pad columns are sliced off downstream.
        U_np = np.pad(U_np, ((0, 0), (0, nb_pad - n_in), (0, nb_pad - n_in)))
    return device_put_process_local(
        U_np, NamedSharding(mesh, band_rotation_spec()))


def _make_rotate_bundle(mesh: Mesh, a_lo: int, nb_active: int, nb_pad: int):
    """Build the jit'd four-copy rotation.  One kernel, one lowering.

    Per copy: slice the active block, put its BAND axis on the mesh axis μ
    does not own, contract against U at :func:`band_mix_spec` for that same
    axis, and constrain the result back to the copy's own ``PSI_*_SPEC``.

    THE OUTPUT CONSTRAINT IS ONE STEP HERE, AND THAT IS NOT AN OVERSIGHT.
    ``qsgw_density.rotate_bands`` needs two (land on 'y', then re-split)
    because its contraction's natural output — n from U's 'y', partial over
    'x' — is not its target layout, and constraining straight to the target
    made XLA all-reduce the full global ψ̃ (3.36 GiB/rank measured, audit
    2026-08-05).  Here the natural output IS the target: n comes from U's
    REPLICATED axis and μ stays on the mesh axis it started on, so the only
    thing left to do is finish the partial sum over the contraction axis, at
    the size of the output shard.  That identity is the whole reason for
    :func:`band_mix_spec`'s replicated n.
    """
    def rot_one(psi_full, U, spec, band_axis):
        axis = _contract_axis(mesh, spec)
        act = jax.lax.slice_in_dim(psi_full, a_lo, a_lo + nb_active,
                                   axis=band_axis)
        if nb_pad != nb_active:
            widths = [(0, 0)] * psi_full.ndim
            widths[band_axis] = (0, nb_pad - nb_active)
            act = jnp.pad(act, widths)
        # ψ's μ axis keeps the mesh axis it already had; the band axis, so
        # far replicated over the free axis, becomes a local sub-slice of
        # data the rank already holds.  MEASURED, because "local" is a claim
        # about the lowering and not only about the layout: XLA lowers this
        # as dynamic-slice + a one-hop collective-permute of the FULLY
        # sharded active tile (source_target_pairs {{0,1},{2,3}} on a 2×2
        # mesh, 6 kB at the gate shape, job 7889407) -- a device-order
        # fix-up, not a gather.  Its payload is nb_act·ns·n_μ/(px·py) per
        # rank, the smallest object in the kernel, and it SHRINKS with P.
        m_spec = _spec_entries(spec, psi_full.ndim)
        m_spec[band_axis] = axis
        act = jax.lax.with_sharding_constraint(
            act, NamedSharding(mesh, P(*m_spec)))
        U_m = jax.lax.with_sharding_constraint(
            U, NamedSharding(mesh, band_mix_spec(axis)))
        if band_axis == 1:
            # band-first (k,m,s,μ): ψ'[k,n,s,μ] = Σ_m U[k,m,n]·ψ[k,m,s,μ]
            out = jnp.einsum('kmn,kmsu->knsu', U_m, act, optimize=True)
        else:
            # band-last (k,s,μ,m): ψ'[k,s,μ,n] = Σ_m ψ[k,s,μ,m]·U[k,m,n]
            out = jnp.einsum('ksum,kmn->ksun', act, U_m, optimize=True)
        # The reduction over m is a psum along ``axis`` ALONE, not a global
        # collective: m is the only index on that axis and every other index
        # of the result is either replicated (n, s) or already placed (μ).
        # Measured on a 2×2 mesh (job 7889407): two all-reduces, replica
        # group 2 not 4, each at exactly one ψ output shard, and XLA's
        # combiner merges the two copies that share a contraction axis into
        # one.  The replicated-U path it replaces had NO collective at all
        # and paid for it with a full (nb, nb) per rank and px-fold
        # redundant flops -- that is the trade, stated rather than implied.
        out = jax.lax.with_sharding_constraint(out, NamedSharding(mesh, spec))
        if nb_pad != nb_active:
            out = jax.lax.slice_in_dim(out, 0, nb_active, axis=band_axis)
        return jax.lax.dynamic_update_slice_in_dim(psi_full, out, a_lo,
                                                   axis=band_axis)

    @jax.jit
    def fn(psi_xn, psi_xr, psi_yr, psi_yn, U):
        U = jnp.asarray(U, dtype=jnp.complex128)
        n_in = int(U.shape[1])
        if n_in != nb_pad:
            U = jnp.pad(U, ((0, 0), (0, nb_pad - n_in), (0, nb_pad - n_in)))
        return tuple(
            rot_one(psi, U, spec, band_axis)
            for psi, (_, spec, band_axis) in zip(
                (psi_xn, psi_xr, psi_yr, psi_yn), _PSI_LAYOUTS))
    return fn


def _rotate_kernel(mesh: Mesh, a_lo: int, nb_active: int, nb_pad: int):
    """One built kernel per (mesh, active window, pad extent).

    ``common.wfn_transforms._cached_jit`` rather than a private dict here:
    it is the tree's one kernel-cache idiom (``qsgw_density`` and the ψ
    transforms both use it), so the caches share a namespace and a
    lifetime.  Imported lazily to keep this module's import graph — jax,
    numpy and ``runtime.padding`` — unchanged for its many importers.
    """
    from common.wfn_transforms import _cached_jit
    return _cached_jit(
        'rotate_wavefunctions', (id(mesh), a_lo, nb_active, nb_pad),
        lambda: _make_rotate_bundle(mesh, a_lo, nb_active, nb_pad))


def rotate_wavefunctions(
    wfns_dft: Wavefunctions,
    U_dft_to_qp_active: jax.Array,
    *,
    enk_active_new: jax.Array,
    enk_base: jax.Array | None = None,
    efermi: float | None,
    mesh_xy: Mesh,
    active_slice: slice | None = None,
) -> Wavefunctions:
    """Return a new ``Wavefunctions`` bundle with the **active subspace**
    rotated by ``U_dft_to_qp_active[k, m, n] = ⟨DFT_m | QP_n⟩``.

    COLUMN CONVENTION.  ``U[k, m, n]`` is component m of QP band n, so
    ψ̃_n = Σ_m U[m,n] ψ_m — ``jnp.linalg.eigh``'s convention, ScaLAPACK's,
    ``qsgw_density.rotate_bands``' and ``sc_iteration._rotate_to_dft_basis``'.
    The transpose is a bug MOST TESTS CANNOT SEE: for a unitary Q, Qᵀ is
    also unitary and also mixes only within the occupied block, so norms,
    orthonormality and occupied-block invariance all survive it.  It is
    pinned in ``tests/multi_device/wfn_rotate_gate.py`` against an explicit
    host-side rotation with a negative control, not against an invariance.

    U IS NEVER REPLICATED
    ---------------------
    Each ψ copy is rotated against U sharded :func:`band_mix_spec` for the
    mesh axis that copy's μ does NOT own — m on 'x' for the two y-copies,
    m on 'y' for the two x-copies.  So no rank holds a full ``(nb, nb)``,
    ψ stays where it is, and the sum over m is a psum along a single mesh
    axis rather than a global collective.  See :func:`band_mix_spec` for
    why n is replicated and :func:`_make_rotate_bundle` for why the output
    needs only one sharding constraint and what the lowering actually
    emits.

    The arithmetic drops by the same factor.  Under a replicated U every
    rank contracted the WHOLE band sum against its own μ slice, so the p
    ranks sharing a μ slice each did the identical work; now each does
    ``nb/p`` of it and the psum adds the pieces.  That redundancy, not the
    bytes, is the larger of the two effects on a square mesh.

    Two U layouts are materialised, not one — the x-copies and the
    y-copies contract on different mesh axes.  What the scaling target
    bounds is the LARGEST U operand any single rank must hold, and that is
    ``nk·nb²/p·16 B`` against ``nk·nb²·16 B`` replicated on every mesh with
    an axis > 1: 6.25 MiB → 0.78 MiB per k at nb=640 on an 8×8 mesh.  Their
    SUM is ``nk·nb²·(1/px + 1/py)·16 B``, a factor px·py/(px+py), and since
    the mesh is square (``common.collectives.resolve_mesh`` refuses others)
    that is p/2: no saving at 2×2, 4× at 8×8 (P=64), 8× at 16×16.  Measured
    at 2×2 (job 7889407): worst operand 2.00×, sum 1.00×.

    What arrives already sharded is the CALLER's half of this.  A
    replicated ``U`` costs nothing extra here (the constraint is a local
    slice) but was replicated before the call; ``sc_iteration`` can hand
    over a distributed one by asking ``_make_kshard_eigh`` for
    ``u_spec=qsgw_density.band_rotation_spec()``, which it already
    supports, or by using ``distributed_eigh_bands``, which emits it.

    Active / inactive partition
    ---------------------------
    The ``active_slice`` (default: ``wfns_dft.slices.sigma`` — the QP
    evaluation window) selects a contiguous band block ``[start, stop)``
    where the QP Hamiltonian has full off-diagonal Σ and we apply the
    band-mixing unitary.  Bands **outside** that window always keep their
    DFT wavefunctions.  Their energies come from ``enk_base`` when one is
    supplied, otherwise from the DFT bundle unchanged.  This lets the SC
    driver apply an energy-only scissor to conduction-sum bands without
    inventing a QP wavefunction rotation for bands the QP matrix never
    touched.

    Validation
    ----------
    Errors if ``active_slice`` extends past ``wfns_dft.slices.sigma`` —
    that would imply we're rotating bands the calculation can't have
    produced QP-basis information for.

    Parameters
    ----------
    wfns_dft
        The original DFT bundle (preserved unchanged across iterations).
    U_dft_to_qp_active
        Per-k unitary on the active block, shape ``(nk, nb_active, nb_active)``.
    enk_active_new
        New eigenvalues on the active block, shape ``(nk, nb_active)``.
    enk_base
        Optional full-band energy ladder, shape ``(nk, nb_full)``.  Only
        entries outside ``active_slice`` survive; active entries are always
        replaced by ``enk_active_new``.  The caller owns preserving logical
        padding in this ladder.  ``None`` takes the exact historical path
        from ``wfns_dft.enk``.
    efermi
        Fermi level; used here to rebuild ``occ`` after both the active and
        optional inactive energy updates.  No energy-only helper owns
        occupations.
    mesh_xy
        2-D device mesh; sharding of the four ψ copies is preserved.
    active_slice
        Contiguous active band block.  Defaults to ``wfns_dft.slices.sigma``.
    """
    sigma_slice = wfns_dft.slices.sigma
    if active_slice is None:
        active_slice = sigma_slice
    a_lo = int(active_slice.start or 0)
    a_hi = int(active_slice.stop)
    s_lo = int(sigma_slice.start or 0)
    s_hi = int(sigma_slice.stop)
    if a_lo < s_lo or a_hi > s_hi:
        raise ValueError(
            f"rotate_wavefunctions: active_slice [{a_lo}, {a_hi}) leaks "
            f"outside the σ-window [{s_lo}, {s_hi}); we have no QP basis "
            f"information for bands beyond the protected window.")
    nb_active = a_hi - a_lo
    if U_dft_to_qp_active.shape[-2:] != (nb_active, nb_active):
        raise ValueError(
            f"rotate_wavefunctions: U shape {U_dft_to_qp_active.shape} "
            f"inconsistent with active block size {nb_active}.")

    # THE CONTRACTED BAND EXTENT MUST DIVIDE THE MESH.  ``nb_active`` is a
    # band WINDOW (b3 - b0), not the loader's mesh-divisible ψ extent, so it
    # need not divide px or py; ``with_sharding_constraint`` refuses an
    # indivisible axis outright rather than degrading (runtime.padding).
    # The divisors come from the spec, not from mesh.shape, so a
    # band_mix_spec that ever became a product axis stays covered.
    pad_div = 1
    for _field, spec, _band_axis in _PSI_LAYOUTS:
        pad_div = math.lcm(pad_div, spec_divisor(
            mesh_xy, band_mix_spec(_contract_axis(mesh_xy, spec)), 1))
    nb_pad = round_up(nb_active, pad_div)

    U = _place_U(U_dft_to_qp_active, mesh_xy, nb_pad)

    # Slice the active block, rotate it against the sharded U, and
    # dynamic-update-slice it back into a copy of the full ψ — one jit for
    # all four copies, so there is a single lowering and no eager pjit per
    # accessor call.  Bands outside the active block stay DFT.
    with mesh_xy:
        rotate = _rotate_kernel(mesh_xy, a_lo, nb_active, nb_pad)
        psi_xn, psi_xr, psi_yr, psi_yn = rotate(
            wfns_dft.psi_xn, wfns_dft.psi_xr, wfns_dft.psi_yr,
            wfns_dft.psi_yn, U)

        # enk: the default branch is the exact historical expression.  The
        # optional base changes inactive ENERGIES only; the active block is
        # overwritten below and the four ψ arrays above always started from
        # wfns_dft.
        if enk_base is None:
            enk_full = wfns_dft.enk.at[:, active_slice].set(
                jnp.asarray(enk_active_new, dtype=wfns_dft.enk.dtype))
        else:
            if tuple(enk_base.shape) != tuple(wfns_dft.enk.shape):
                raise ValueError(
                    f"rotate_wavefunctions: enk_base shape {enk_base.shape} "
                    f"does not match full DFT energies "
                    f"{wfns_dft.enk.shape}.")
            enk_full = jnp.asarray(
                enk_base, dtype=wfns_dft.enk.dtype).at[:, active_slice].set(
                    jnp.asarray(enk_active_new, dtype=wfns_dft.enk.dtype))
        rep2 = NamedSharding(mesh_xy, P(None, None))
        enk_full = jax.lax.with_sharding_constraint(enk_full, rep2)
        occ_full = jax.lax.with_sharding_constraint(
            _build_occ(enk_full, wfns_dft.slices, efermi), rep2)

    return Wavefunctions(
        psi_xn=psi_xn, psi_xr=psi_xr, psi_yr=psi_yr, psi_yn=psi_yn,
        enk=enk_full, occ=occ_full, slices=wfns_dft.slices,
    )


# ---------------------------------------------------------------------------
# Band-basis projection — Σ_mn(k) = Σ_{s,μ,s',μ'} ψ*_m(s,μ) Σ(s,μ,s',μ') ψ_n(s',μ')
#
# Lives here because the only state these contractions need is the (xr, yn)
# pair of sharded ψ copies that the bundle owns; consumers (cohsex_sigma,
# the AOT memory model) operate at the bundle's seam.
# ---------------------------------------------------------------------------

def project(psi_xr, psi_yn, sigma_k, *, layout='legacy', mesh_xy=None,
           face_shape=None, face_project_fn=None):
    """Σ(nk, s, μ, s, μ) → Σ(nk, m, n) in band basis.

    ``layout='legacy'`` (default): the exact body this function has always
    had — ``psi_xr``/``psi_yn`` are the legacy bundle fields.  UNCHANGED
    by this parameter's addition (dead code eliminated at trace time for
    every existing caller, none of which pass ``layout``).

    ``layout='face'``: this is the "older COHSEX facade" the audit report
    names (§5) — it now ROUTES rather than re-implements, through
    ``common.contract_bands.contract_bands_block_reshard(layout='face')``,
    the single owner of both the legacy and face band projection.
    ``psi_xr``/``psi_yn`` are then actually ``psi_nmu``/``psi_mun`` (same
    two argument SLOTS, same meaning across layouts: first operand
    conjugated inside, second used as-is — see that module's docstring).
    Pass a pre-built ``face_project_fn`` (from the SAME factory call, e.g.
    built once inside a kernel factory alongside ``_Gv_fftn`` et al.) to
    avoid rebuilding — and re-warming — two ``distrib_la.gemm_plan``s on
    every call; ``mesh_xy``/``face_shape`` are the fallback that builds
    one inline (correct but wasteful if called repeatedly).
    """
    if layout == 'legacy':
        left = jnp.einsum('kmsx,ksxty->kmty',
                          jnp.conj(psi_xr), sigma_k, optimize=True)
        return jnp.einsum('kmty,ktyn->kmn', left, psi_yn, optimize=True)
    if layout != 'face':
        raise ValueError(
            f"project: layout must be 'legacy' or 'face', got {layout!r}")
    fn = face_project_fn
    if fn is None:
        if mesh_xy is None or face_shape is None:
            raise ValueError(
                "project(layout='face') requires either face_project_fn=, "
                "or both mesh_xy= and face_shape= to build one inline "
                "(see common.contract_bands.contract_bands_block_reshard)")
        from common.contract_bands import contract_bands_block_reshard
        fn = contract_bands_block_reshard(
            mesh_xy, layout='face', face_shape=face_shape)
    return fn(psi_xr, sigma_k, psi_yn)


def project_ri(psi_xr, psi_yn, sigma_k):
    """Σ(nk, s, μ, s, μ) → (2, nk, m, n) with [Re, Im] channels.

    Used by the windowed PPM Σ^c(ω) τ-loop, where the crossing window
    keeps only ``Im[coeff·σ^τ]`` so σ^τ has to carry both channels.
    A sharded reduce-scatter variant lives in ``ppm_sigma`` for the
    multi-device path.
    """
    sigma_ri = jnp.stack((jnp.real(sigma_k), jnp.imag(sigma_k)), axis=0)
    left = jnp.einsum('kmsx,cksxty->ckmty',
                      jnp.conj(psi_xr), sigma_ri, optimize=True)
    return jnp.einsum('ckmty,ktyn->ckmn',
                      left, psi_yn, optimize=True).astype(jnp.complex128)
