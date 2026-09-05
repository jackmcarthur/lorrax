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

from common.wfn_layout import PSI_MUN_SPEC, PSI_NMU_SPEC
from file_io.wfn_basis import WavefunctionBasisReceipt
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
    #: [b2-b0, b4-b0) — conduction bands over the PADDED loaded carrier.
    #: This is a shape/layout slice, not a physical spectral window.
    #:
    #: The physical union is :attr:`cond_all_logical`, which ends at
    #: ``b4_logical``.  Keeping both names on this carrier prevents a consumer
    #: from recovering a logical count from a padded shape.
    cond_all: slice = slice(0, 0)
    #: Logical union top used by spectral consumers.  On a split deck this is
    #: max(chi, sigma); on every deck it excludes the process-mesh pad in b4.
    b4_logical: int = 0

    @classmethod
    def from_band_edges(cls, b0: int, b1: int, b2: int, b3: int, b4: int,
                        *, b4_chi: int | None = None,
                        b4_sigma: int | None = None,
                        b4_logical: int | None = None) -> BandSlices:
        if not (b0 <= b1 <= b2 <= b3 <= b4):
            raise ValueError(f"Invalid band edges: {(b0, b1, b2, b3, b4)}")
        nb_full = b4 - b0
        b4_chi = b4 if b4_chi in (None, 0) else int(b4_chi)
        b4_sigma = b4 if b4_sigma in (None, 0) else int(b4_sigma)
        b4_logical = b4 if b4_logical in (None, 0) else int(b4_logical)
        if not (b2 <= b4_logical <= b4):
            raise ValueError(
                f"Invalid b4_logical={b4_logical}: the physical loaded-band "
                f"top must satisfy b2={b2} <= b4_logical <= padded b4={b4}.")
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
            b4_logical=b4_logical,
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
    def cond_all_logical(self) -> slice:
        """Conduction bands in the logical union of the chi and Sigma sums.

        This is the minimax tau-axis window: an interval built from ``cond``
        can under-cover Sigma when Sigma is larger, while one built from
        padded ``cond_all`` changes with process geometry.
        """
        return slice(self.b2 - self.b0, self.b4_logical - self.b0)

    @property
    def nb_full_logical(self) -> int:
        """Logical loaded-band count, excluding process-mesh padding."""
        return self.b4_logical - self.b0

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

# The two FACE specs are imported from ``common.wfn_layout`` and re-exported
# here for the established GW public API.  See the module docstring and
# reports/gwjax_low_mem_bands_audit_2026-08-22/report.md for why both
# orientations are required by cuBLASMp's portable N,N-only path.

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
class ParentGreenCarrier:
    """Orbit-packed raw-parent operands for Green-function contractions."""

    psi_nmu: jax.Array
    psi_mun: jax.Array
    enk: jax.Array
    occ: jax.Array
    plan: object

    @functools.partial(jax.jit, static_argnames=('bands',))
    def band_mask(self, bands: slice) -> jax.Array:
        nb = int(self.enk.shape[1])
        lo = int(bands.start or 0)
        hi = int(bands.stop if bands.stop is not None else nb)
        row = (jnp.arange(nb) >= lo) & (jnp.arange(nb) < hi)
        return jnp.broadcast_to(row[None, :], self.enk.shape)


jax.tree_util.register_dataclass(
    ParentGreenCarrier,
    data_fields=['psi_nmu', 'psi_mun', 'enk', 'occ'],
    meta_fields=['plan'],
)


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
    #: Optional raw-parent, orbit-packed operands.  This is an acceleration
    #: carrier only; all observable/runtime operators remain in the primary
    #: bundle's canonical full-k basis until their own packed seam lands.
    green_parent: ParentGreenCarrier | None = None
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
# Provenance is intentionally not a Wavefunctions field: the orchestration
# owner keeps its receipt beside this numerical carrier and validates it
# before array extraction.  Per-WFN identities therefore cannot enter a
# treedef, and a compiled carrier round-trip cannot silently discard one.
jax.tree_util.register_dataclass(
    Wavefunctions,
    data_fields=['psi_xn', 'psi_xr', 'psi_yr', 'psi_yn',
                 'psi_nmu', 'psi_mun', 'green_parent', 'enk', 'occ'],
    meta_fields=['slices', 'layout'],
)


@functools.partial(
    jax.jit, static_argnames=("state_bands", "projection_bands"))
def projected_state_amplitude_envelope(
    wfns: Wavefunctions,
    *,
    state_bands: slice,
    projection_bands: slice,
) -> jax.Array:
    """State weights from the actual Sigma band-projection carriers.

    For one intermediate state ``n``, the ISDF Green function contains the
    outer product of that state's left/right centroid wavefunctions.  The
    final Sigma element contains the corresponding left/right projection
    carriers from the requested output subspace
    (``docs/theory/physics.md``, section 5).  Cauchy--Schwarz gives the
    per-k state-side projected matrix-element envelope

    ``A_kn = ||psi^L_kn|| ||psi^R_kn||
             max_i||psi^L_ki|| max_j||psi^R_kj||``.

    The state factors are returned for ``state_bands``; the projection maxima
    are measured on ``projection_bands``.  All four legacy carriers, or both
    orientations of the face carrier, therefore enter with their actual
    values instead of assuming the nominally equivalent copies are identical.
    Thus planner mass follows the real requested matrix-element projection
    instead of assigning unit amplitude to every state.  It remains an
    operator envelope: spatial pole phases and cancellation are deliberately
    not converted into a claim of physical relative Sigma accuracy.

    Parameters
    ----------
    wfns : Wavefunctions
        Canonical wavefunction carrier in either legacy or face layout.
    state_bands : slice
        Intermediate-state band range used by the Green function.
    projection_bands : slice
        Requested Sigma output band range.

    Returns
    -------
    jax.Array, shape (nk, n_state_bands)
        Nonnegative projected state-amplitude envelopes.  Reductions over
        centroid axes retain the carrier's named sharding semantics.
    """
    if wfns.layout == "legacy":
        state_left = jnp.sqrt(jnp.sum(
            jnp.abs(wfns.xn(state_bands)) ** 2, axis=(1, 2)))
        state_right = jnp.sqrt(jnp.sum(
            jnp.abs(wfns.yr(state_bands)) ** 2, axis=(2, 3)))
        projection_left = jnp.max(jnp.sqrt(jnp.sum(
            jnp.abs(wfns.xr(projection_bands)) ** 2, axis=(2, 3))), axis=1)
        projection_right = jnp.max(jnp.sqrt(jnp.sum(
            jnp.abs(wfns.yn(projection_bands)) ** 2, axis=(1, 2))), axis=1)
        return (state_left * state_right
                * projection_left[:, None] * projection_right[:, None])

    # Face-layout ψ cannot be sliced at an arbitrary band boundary.  Reduce
    # its full two orientations first, use the canonical small band mask for
    # the projection maximum, and slice only the resulting (nk, nb) weights.
    left_norm = jnp.sqrt(jnp.sum(jnp.abs(wfns.psi_mun) ** 2, axis=(1, 2)))
    right_norm = jnp.sqrt(jnp.sum(jnp.abs(wfns.psi_nmu) ** 2, axis=(2, 3)))
    projection_mask = wfns.band_mask(projection_bands)
    projection_left = jnp.max(
        jnp.where(projection_mask, right_norm, 0.0), axis=1)
    projection_right = jnp.max(
        jnp.where(projection_mask, left_norm, 0.0), axis=1)
    state_weight = (left_norm * right_norm
                    * projection_left[:, None] * projection_right[:, None])
    return state_weight[:, state_bands]


@dataclass(frozen=True)
class AuthenticatedWavefunctions:
    """Host orchestration binding of a numerical carrier to its receipt.

    This type is intentionally not a JAX pytree.  Orchestration validates and
    unwraps it before a compiled call; an accidental attempt to pass the
    authenticated object through ``jax.jit`` therefore refuses instead of
    dropping provenance or specializing on per-WFN hashes.
    """

    wavefunctions: Wavefunctions
    receipt: WavefunctionBasisReceipt

    def __post_init__(self) -> None:
        # A transverse binding names the LorentzCarriers container (one
        # object whether the three labels share a carrier or not), so the
        # identity check in w_isdf compares against what Sigma consumes.
        if not isinstance(self.wavefunctions, (Wavefunctions, LorentzCarriers)):
            raise TypeError(
                "AuthenticatedWavefunctions requires a Wavefunctions "
                f"carrier; got {type(self.wavefunctions).__name__}")
        if not isinstance(self.receipt, WavefunctionBasisReceipt):
            raise TypeError(
                "AuthenticatedWavefunctions requires a canonical basis "
                f"receipt; got {type(self.receipt).__name__}")
        self.receipt.assert_matches_carrier(
            self.wavefunctions, where="AuthenticatedWavefunctions")


def face_kernel_kwargs(wfns: "Wavefunctions", wfns_right=None) -> dict:
    """``{}`` under ``layout='legacy'``; ``{"layout": "face", "face_shape":
    (nk, nb_full, n_rmu, nspinor)}`` under ``layout='face'``.  With a
    distinct right endpoint, also returns ``right_face_shape`` when its
    centroid extent differs.

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
    if wfns_right is None:
        wfns_right = wfns
    if wfns.layout != wfns_right.layout:
        raise ValueError(
            "face_kernel_kwargs: endpoint layouts differ: "
            f"{wfns.layout!r} vs {wfns_right.layout!r}")
    if wfns.layout != "face":
        return {}
    nk, s, mu, _ = wfns.psi_mun.shape
    nk_r, s_r, mu_r, _ = wfns_right.psi_mun.shape
    left_shape = (nk, wfns.slices.nb_full, mu, s)
    right_shape = (nk_r, wfns_right.slices.nb_full, mu_r, s_r)
    result = {"layout": "face", "face_shape": left_shape}
    if right_shape != left_shape:
        result["right_face_shape"] = right_shape
    return result


def green_face_kernel_kwargs(wfns: "Wavefunctions") -> dict:
    """Face G shape plus optional raw-parent transport plan.

    Projection factories continue to use :func:`face_kernel_kwargs`, whose
    k extent is full BZ.  This separate seam exists because only the Green
    band contraction moves to raw parents; its completed operator returns to
    full k before any FFT or projection sees it.
    """
    result = face_kernel_kwargs(wfns)
    parent = wfns.green_parent
    if not result or parent is None:
        return result
    nk, ns, nmu, nb = (int(v) for v in parent.psi_mun.shape)
    if tuple(int(v) for v in parent.psi_nmu.shape) != (nk, nb, ns, nmu):
        raise ValueError(
            "green_face_kernel_kwargs: parent face orientations disagree: "
            f"{parent.psi_mun.shape}/{parent.psi_nmu.shape}.")
    return {
        "layout": "face",
        "face_shape": (nk, nb, nmu, ns),
        "k_unfold_plan": parent.plan,
    }


def pack_parent_green_faces(
    psi_rmu_Y_parent, psi_rmuT_X_parent, *, plan, mesh_xy: Mesh,
) -> tuple[jax.Array, jax.Array]:
    """Convert raw-parent loader outputs to the orbit-packed face layout."""
    with mesh_xy:
        psi_nmu = jax.lax.with_sharding_constraint(
            psi_rmu_Y_parent, NamedSharding(mesh_xy, PSI_NMU_SPEC))
        psi_mun = jax.lax.with_sharding_constraint(
            jnp.conj(psi_rmuT_X_parent).transpose(0, 3, 1, 2),
            NamedSharding(mesh_xy, PSI_MUN_SPEC))
        return plan.pack_face_pair(psi_nmu, psi_mun)


def build_packed_parent_green_carrier(
    wfns: "Wavefunctions", psi_nmu_parent, psi_mun_parent, *, plan,
    mesh_xy: Mesh,
) -> ParentGreenCarrier:
    """Bind packed raw-parent faces to the full-k bundle's scalar tables."""
    if wfns.layout != "face":
        raise ValueError(
            "build_packed_parent_green_carrier requires layout='face'; "
            "the legacy bundle retains its established full-k local GEMMs.")
    expected_nmu = (
        int(plan.n_parent), int(wfns.slices.nb_full), int(plan.nspinor),
        int(plan.n_centroid_packed))
    expected_mun = (
        int(plan.n_parent), int(plan.nspinor),
        int(plan.n_centroid_packed), int(wfns.slices.nb_full))
    if tuple(int(v) for v in psi_nmu_parent.shape) != expected_nmu:
        raise ValueError(
            "build_packed_parent_green_carrier: psi_nmu shape "
            f"{psi_nmu_parent.shape} != {expected_nmu}.")
    if tuple(int(v) for v in psi_mun_parent.shape) != expected_mun:
        raise ValueError(
            "build_packed_parent_green_carrier: psi_mun shape "
            f"{psi_mun_parent.shape} != {expected_mun}.")
    with mesh_xy:
        psi_nmu = jax.lax.with_sharding_constraint(
            psi_nmu_parent, NamedSharding(mesh_xy, PSI_NMU_SPEC))
        psi_mun = jax.lax.with_sharding_constraint(
            psi_mun_parent, NamedSharding(mesh_xy, PSI_MUN_SPEC))
        enk = plan.parent_rows(wfns.enk)
        occ = plan.parent_rows(wfns.occ)
        rep2 = NamedSharding(mesh_xy, P(None, None))
        enk = jax.lax.with_sharding_constraint(enk, rep2)
        occ = jax.lax.with_sharding_constraint(occ, rep2)
    return ParentGreenCarrier(
        psi_nmu=psi_nmu, psi_mun=psi_mun, enk=enk, occ=occ, plan=plan)


def attach_packed_parent_green_carrier(
    wfns: "Wavefunctions", psi_nmu_parent, psi_mun_parent, *, plan,
    mesh_xy: Mesh,
) -> "Wavefunctions":
    """Attach already-packed raw-parent faces to a full-k face bundle."""
    carrier = build_packed_parent_green_carrier(
        wfns, psi_nmu_parent, psi_mun_parent, plan=plan, mesh_xy=mesh_xy)
    import dataclasses
    return dataclasses.replace(wfns, green_parent=carrier)


def attach_parent_green_carrier(
    wfns: "Wavefunctions", psi_rmu_Y_parent, psi_rmuT_X_parent, *, plan,
    mesh_xy: Mesh,
) -> "Wavefunctions":
    """Pack and attach raw-parent loader outputs to a face bundle."""
    psi_nmu, psi_mun = pack_parent_green_faces(
        psi_rmu_Y_parent, psi_rmuT_X_parent, plan=plan, mesh_xy=mesh_xy)
    return attach_packed_parent_green_carrier(
        wfns, psi_nmu, psi_mun, plan=plan, mesh_xy=mesh_xy)


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


class LorentzCarriers:
    """The spatial-current carriers: one :class:`Wavefunctions` per Lorentz
    label ``mu_L`` in {1, 2, 3}, on the transverse centroid set.

    Under the shipped ``sigma.p`` lift the three labels ride ONE
    four-spinor, so all three entries are the same object (no copies);
    under the per-channel velocity balance (``common.bispinor_init``,
    ``bispinor_current_balance = velocity``) each label has its own carrier,
    and every Green's-function endpoint or Sigma bra/ket for label ``mu_L``
    must be drawn from ``channel(mu_L)``.  That endpoint rule is the whole
    of the two-carrier bookkeeping (docs/architecture/four_current_wiring.md).

    Attribute reads fall through to channel 1 so extent, layout and band
    window readers (``padded_centroid_extent``, ``.layout``, ``.slices``,
    ``psi_inventory_bytes``) need no special case: those are properties every
    channel shares.  A block builder that wants an endpoint asks for it.
    """

    def __init__(self, channels):
        chans = tuple(channels)
        if len(chans) != 3:
            raise ValueError(
                f"LorentzCarriers needs exactly three channel bundles; got "
                f"{len(chans)}")
        object.__setattr__(self, "_channels", chans)

    @classmethod
    def shared(cls, wfns: "Wavefunctions") -> "LorentzCarriers":
        """All three labels on one bundle (the shipped sigma.p carrier)."""
        return cls((wfns, wfns, wfns))

    @property
    def channels(self) -> tuple:
        return self._channels

    @property
    def one_carrier(self) -> bool:
        """True when the three labels are literally the same bundle."""
        first = self._channels[0]
        return all(c is first for c in self._channels)

    def channel(self, mu_L: int) -> "Wavefunctions":
        mu_L = int(mu_L)
        if mu_L not in (1, 2, 3):
            raise ValueError(
                f"LorentzCarriers.channel: mu_L must be 1, 2 or 3; got {mu_L}")
        return self._channels[mu_L - 1]

    def __getattr__(self, name):
        # Only reached for names not found on the container itself.
        return getattr(self._channels[0], name)

    def __repr__(self) -> str:
        tag = "one carrier" if self.one_carrier else "three carriers"
        return f"LorentzCarriers({tag}, layout={self._channels[0].layout!r})"


def as_lorentz_carriers(wfns) -> "LorentzCarriers | None":
    """Accept a bare bundle (shared across labels) or a LorentzCarriers."""
    if wfns is None or isinstance(wfns, LorentzCarriers):
        return wfns
    return LorentzCarriers.shared(wfns)


def endpoint_bundles(left: "Wavefunctions", right: "Wavefunctions",
                     mu_L: int, nu_L: int):
    """Operands for a Sigma block whose two endpoints ride different carriers.

    Returns ``(proj, g)``: ``proj`` supplies the OUTER projection (bra from
    ``left``, ket from ``right``) and ``g`` the G-build's two operands with
    ``gamma~^{mu_L}`` folded into the direct field (from ``left``) and
    ``gamma~^{nu_L}`` into the conjugated one (from ``right``).  Under
    ``layout='legacy'`` the four fields are independent arrays, so ONE mixed
    bundle serves both roles and ``g`` is ``None``; under ``layout='face'``
    the same two arrays serve both roles and the two objects differ (this is
    the reason ``cohsex_sigma``'s face ``sigma_sx`` has ``wfns_g``).

    ``left is right`` reproduces :func:`with_lorentz_vertices` byte for
    byte: ``(left, with_lorentz_vertices(left, mu_L, nu_L))`` on face and
    ``(with_lorentz_vertices(left, mu_L, nu_L), None)`` on legacy.
    """
    import dataclasses
    if left.layout != right.layout:
        raise ValueError(
            "endpoint_bundles: the two endpoint carriers must share a "
            f"layout; got {left.layout!r} and {right.layout!r}")
    if left.slices != right.slices:
        raise ValueError(
            "endpoint_bundles: the two endpoint carriers must share a band "
            "window; their BandSlices differ")
    if left is right:
        g = with_lorentz_vertices(left, mu_L, nu_L)
        return (left, g) if left.layout == "face" else (g, None)
    left_g = with_lorentz_vertices(left, mu_L, 0)
    right_g = with_lorentz_vertices(right, 0, nu_L)
    if left.layout == "face":
        proj = dataclasses.replace(left, psi_mun=right.psi_mun)
        g = dataclasses.replace(left_g, psi_nmu=right_g.psi_nmu)
        return proj, g
    mixed = dataclasses.replace(
        left_g, psi_yr=right_g.psi_yr, psi_yn=right.psi_yn)
    return mixed, None


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


def padded_centroid_extent(wfns: "Wavefunctions") -> int:
    """PADDED centroid (μ) extent of ``wfns``, whichever field carries it.

    ``psi_yr``'s trailing axis under ``'legacy'``, ``psi_mun``'s μ axis
    (index 2) under ``'face'`` — the SAME logical quantity
    (``load_centroids_band_chunked``'s mesh-padded ``n_rmu``), carried by a
    different field per layout.  The ONE owner of that read: the packed
    photon response, the sixteen-block photon Σ and the bare Σ^B all pad
    their on-disk V tiles up to this extent and must agree on it.
    """
    if wfns.layout == "legacy":
        return int(wfns.psi_yr.shape[-1])
    if wfns.layout == "face":
        return int(wfns.psi_mun.shape[2])
    raise ValueError(
        f"padded_centroid_extent: unknown layout {wfns.layout!r}")


def bundle_bytes_per_rank(wfns: "Wavefunctions") -> dict:
    """MEASURED (never modeled) per-rank byte residency of this bundle's
    ψ fields, read off each array's OWN ``.addressable_shards`` — the
    same instrument ``docs/architecture/zeta_fit_face_psi_cct.md``'s own
    verification section prefers over a sharding-blind global-shape read
    (``KNOWN_LORRAX_ISSUES.md``'s ``mem_probe`` row: a global shape
    cannot tell a genuinely 2-D-sharded array from a same-shape
    single-axis one).

    Returns ``{field: bytes, ..., 'total': bytes}`` over
    :func:`psi_field_names` for this bundle's own ``layout``.  When a
    raw-parent Green carrier is attached, its two ψ faces are included under
    ``green_parent.psi_nmu``/``green_parent.psi_mun``; the small replicated
    energy/occupation tables follow the established convention and are not
    part of this ψ inventory.

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
    if wfns.green_parent is not None:
        for f in ("psi_nmu", "psi_mun"):
            arr = getattr(wfns.green_parent, f)
            out[f"green_parent.{f}"] = int(sum(
                int(s.data.nbytes) for s in arr.addressable_shards))
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
    basis_receipt=None,
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

    wfns = Wavefunctions(
        psi_xn=psi_xn, psi_xr=psi_xr, psi_yr=psi_yr, psi_yn=psi_yn,
        enk=enk_full, occ=occ_full, slices=slices)
    if basis_receipt is not None:
        basis_receipt.assert_matches_carrier(
            wfns, where="build_wavefunctions")
    return wfns


def build_wavefunctions_face(
    psi_rmu_Y, psi_rmuT_X, *, enk_full, slices, mesh_xy, efermi=None,
    basis_receipt=None,
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

    wfns = Wavefunctions(
        psi_nmu=psi_nmu, psi_mun=psi_mun,
        enk=enk_full, occ=occ_full, slices=slices, layout="face")
    if basis_receipt is not None:
        basis_receipt.assert_matches_carrier(
            wfns, where="build_wavefunctions_face")
    return wfns


def wavefunctions_face_from_restart(
    psi_nmu, psi_mun, *, enk_full, slices, mesh_xy, efermi=None,
    basis_receipt=None,
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

    wfns = Wavefunctions(
        psi_nmu=psi_nmu, psi_mun=psi_mun,
        enk=enk_full, occ=occ_full, slices=slices, layout="face")
    if basis_receipt is not None:
        basis_receipt.assert_matches_carrier(
            wfns, where="wavefunctions_face_from_restart")
    return wfns


# ---------------------------------------------------------------------------
# pack_band_window — the canonical distributed band-window repack
#
# The face carrier's psi_mun/psi_nmu span the FULL [b0,b4) loaded extent and
# cannot legally be SLICED to an arbitrary band sub-range (module docstring;
# report obstacle #3): the band axis is mesh-sharded and a logical window
# need not be mesh-divisible.  Two bring-up substitutes already exist for
# this --- Wavefunctions.band_mask (a boolean weight, consumed by
# greens_function_kernel.build_G_tau's mask= seam) and the same "weight,
# don't window" idiom inlined in ppm_tau_kernel._bracketed_face --- and
# BOTH accept the cost the mask docstring names outright: "every masked
# face call pays the FULL nb_full GEMM, not the windowed one."  This is
# the "future canonical distributed pack_band_window" that docstring
# forward-references: a ONE-TIME distributed repack that actually shrinks
# the resident band extent, so a bracket's G-build GEMM runs at its own
# TRUE width (owner directive: "the gemm [should] be partial, or
# materialize the band slices as their own array copies at the start of
# the sigma procedure" --- this is the second half of that sentence).
# ---------------------------------------------------------------------------

def pack_band_window(
    wfns: "Wavefunctions", lo: int, hi: int | None, *, mesh_xy: Mesh,
) -> tuple[jax.Array, jax.Array]:
    """Repack the face carrier down to a compact pair holding ONLY bands
    ``[lo, hi)``, band axis padded to the mesh multiple with ZERO columns.

    Returns ``(psi_mun_w, psi_nmu_w)`` --- the SAME two field names/specs
    as the full carrier (:data:`PSI_MUN_SPEC` / :data:`PSI_NMU_SPEC`), just
    at a smaller (padded) band extent, so a caller that already knows how
    to build a ``distrib_la.gemm_plan``/``build_G`` at ``k=w_pad`` instead
    of ``k=nb_full`` needs no OTHER change to consume it (``greens_
    function_kernel._face_build_G``'s own operand contract).

    THE PAD IS LOGICAL METADATA, NEVER PHYSICS.  The ``w_pad - width``
    trailing columns are exact-zero ψ, born dead the same way every other
    mesh pad band in this codebase is (``BandSlices``' own ``b4`` pad,
    ``runtime.padding``): they contribute exactly zero to any BILINEAR
    contraction (the G-build, the CCT Gram, every consumer this carrier
    has) and must never be treated as real modes by anything that is NOT
    bilinear in ψ --- an eigensolve or an occupation sum fed by this
    window's output would need its own explicit strip first (report §3's
    own warning, restated: "Zero-weighted bands are NOT automatically safe
    for an eigensolve or an occupation sum").  This function only
    guarantees the zero pad for a G/T-style contraction; it is not a
    general-purpose band-window primitive for every consumer shape.

    Mechanism: :func:`jax.lax.slice_in_dim` on the mesh-sharded band axis
    followed by a :func:`jax.lax.with_sharding_constraint` back onto the
    canonical face spec.  Both operands' band axis is genuinely mesh-
    sharded going IN (unlike ``rotate_wavefunctions``' legacy ψ copies,
    whose band axis starts REPLICATED and whose slice-then-constrain is
    therefore a free local op) --- slicing ``[lo, hi)`` at bounds that need
    not align to either mesh axis's shard boundary is a genuine one-time
    ``O(S·width/nb_full)``-class data movement (JAX's own SPMD partitioner
    resolves the collective; it is bounded by the SLICED width because the
    slice is lowered before the resharding constraint, not by the full
    resident ``nb_full`` extent --- verified by HLO trace on real 4-rank
    CUDA, see ``tests/test_pack_band_window.py``).  This is a ONE-TIME
    repack, meant to run ONCE per (wfns, bracket) at Sigma-procedure start
    and be reused across every τ node / ω window that bracket integrates
    --- never inside a per-τ hot loop, the same discipline every
    ``distrib_la.gemm_plan`` call site in this codebase already follows.

    THE TRIVIAL FULL-WINDOW FAST PATH.  ``lo=0, hi=nb_full`` (or
    ``hi=None``) is the ordinary, un-bracketed GN-PPM case --- "the
    production head=full/gn_ppm deck this port targets runs the trivial
    single-bracket plan" (``ppm_tau_kernel._get_sigma_kij_kernel``'s own
    docstring).  ``nb_full`` is ALREADY mesh-divisible (``BandSlices.b4``'s
    own invariant), so that window is already a legal face matrix and
    packing it would be a real collective bought for nothing.  This
    function detects exactly that case and returns the RESIDENT carrier
    arrays unchanged --- no new array, no collective, zero cost.

    Refuses (does not guess) an illegal or degenerate window rather than
    clamping it, and refuses a non-square mesh rather than silently
    under-padding one axis --- see the two checks below for the exact
    conditions.
    """
    if wfns.layout != "face":
        raise ValueError(
            f"pack_band_window: wfns.layout={wfns.layout!r}, must be "
            f"'face' --- the legacy carrier already stores exactly the "
            f"physical slice per accessor (.xn()/.xr()/.yr()/.yn()) and "
            f"has nothing to pack.")
    nb_full = int(wfns.slices.nb_full)
    lo = int(lo)
    hi_ = nb_full if hi is None else int(hi)
    if not (0 <= lo < hi_ <= nb_full):
        raise ValueError(
            f"pack_band_window: window [{lo}, {hi_}) is not a legal, "
            f"non-empty, contiguous sub-range of the loaded extent "
            f"[0, {nb_full}) --- refusing rather than clamping or "
            f"guessing.  A caller wanting the UNION of several disjoint "
            f"ranges must pack and consume each one separately (exactly "
            f"how ppm_tau_kernel's bracket loop already treats brackets "
            f"as independent contiguous windows); this primitive has no "
            f"multi-range form.  got: lo={lo}, hi={hi_}, nb_full={nb_full}.")
    px = int(mesh_xy.shape["x"])
    py = int(mesh_xy.shape["y"])
    if px != py:
        raise ValueError(
            f"pack_band_window: mesh is {px}x{py}, not square --- every "
            f"low_mem_bands face routine assumes a square (x,y) grid "
            f"(common.collectives.resolve_mesh's own standing invariant; "
            f"CLAUDE.md: 'Almost all operations are done on a square 2d "
            f"processor grid'); a single padding divisor below would "
            f"silently under-pad whichever axis is smaller on a "
            f"rectangular mesh rather than refusing the unsupported "
            f"geometry by name.")
    q = px
    width = hi_ - lo
    w_pad = round_up(width, q)
    if lo == 0 and hi_ == nb_full:
        # nb_full % q == 0 is BandSlices' own invariant (the world-size pad
        # on b4), so w_pad == width == nb_full here always; the window IS
        # the resident carrier.  Reuse it verbatim.
        return wfns.psi_mun, wfns.psi_nmu

    pack = _pack_band_window_kernel(mesh_xy, lo, hi_, w_pad)
    return pack(wfns.psi_mun, wfns.psi_nmu)


def _make_pack_band_window(mesh_xy: Mesh, lo: int, hi_: int, w_pad: int):
    width = hi_ - lo
    pad_mun = ((0, 0), (0, 0), (0, 0), (0, w_pad - width))
    pad_nmu = ((0, 0), (0, w_pad - width), (0, 0), (0, 0))

    @jax.jit
    def pack(psi_mun, psi_nmu):
        mun_w = jax.lax.slice_in_dim(psi_mun, lo, hi_, axis=3)
        nmu_w = jax.lax.slice_in_dim(psi_nmu, lo, hi_, axis=1)
        if w_pad != width:
            mun_w = jnp.pad(mun_w, pad_mun)
            nmu_w = jnp.pad(nmu_w, pad_nmu)
        mun_w = jax.lax.with_sharding_constraint(
            mun_w, NamedSharding(mesh_xy, PSI_MUN_SPEC))
        nmu_w = jax.lax.with_sharding_constraint(
            nmu_w, NamedSharding(mesh_xy, PSI_NMU_SPEC))
        return mun_w, nmu_w
    return pack


def _pack_band_window_kernel(mesh_xy: Mesh, lo: int, hi_: int, w_pad: int):
    """One compiled repack kernel per ``(mesh, lo, hi, w_pad)`` --- the
    same ``common.wfn_transforms._cached_jit`` idiom :func:`_rotate_kernel`
    uses, so a caller that revisits the same bracket window (a second SC
    iteration, a second Σ branch over the same plan) does not re-trace."""
    from common.wfn_transforms import _cached_jit
    return _cached_jit(
        'pack_band_window', (id(mesh_xy), lo, hi_, w_pad),
        lambda: _make_pack_band_window(mesh_xy, lo, hi_, w_pad))


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


#: Cache for the face-layout rotation kernel — keyed like ``_rotate_kernel``
#: (mesh, active window) plus the face carrier's own static extents
#: (nb_full, n_rmu, ns, nk).  Holds the jit'd ``fn`` AND the two
#: ``distrib_la.GemmPlan``s it closes over, so both are built/warmed once
#: per (mesh, shape) and reused across every SC iteration.
_FACE_ROTATE_CACHE: dict = {}


def _place_U_face(U, mesh_xy: Mesh):
    """Face-layout counterpart of :func:`_place_U`'s array-kind dispatch —
    called OUTSIDE any ``jax.jit`` (Python-level, same call site shape as
    legacy's own ``U = _place_U(...)`` before ``rotate(...)``), for the
    SAME reason: distinguishing a genuine host-numpy ``U`` from an
    already-placed ``jax.Array`` is only meaningful before a jit
    boundary — every argument reaching an already-traced body is a
    tracer regardless of what it started as, so this dispatch would be
    dead code if performed inside :func:`_face_rotate_kernel`'s jit'd
    ``fn`` (a real defect class TASTE rule 3 names: an implicit
    multi-process ``device_put`` at the jit argument list).

    Differs from :func:`_place_U` in ONE respect: a host array is placed
    fully REPLICATED (``P(None,None,None)``), not at
    ``qsgw_density.band_rotation_spec()``.  ``_place_U``'s target assumes
    its caller has already rounded the active window to a mesh-divisible
    ``nb_pad``; here ``nb_active`` is a raw σ-window size with no such
    guarantee (see :func:`_face_embed_active_U`), so replicated is the
    only ALWAYS-legal placement — free to reshard afterward regardless
    (a local slice from a replicated input, per ``_place_U``'s own
    comment for that case).

    BYTE FIGURE, stated per TASTE.md rule 1 ("replication whose extent is
    set by a bounded dimension — N_b at the sigma window — is a judgment
    call and must carry its byte figure at the site"): this replicates
    ``nk·nb_active²`` complex128, i.e. ``nk·nb_active²·16`` bytes PER
    RANK — an N_b-class object (the active σ-window, bounded by the
    deck's own band count), never an N_μ-class one.  At nb_active=640,
    nk=144 this is 900.0 MiB/rank (144·640²·16 bytes — corrected
    2026-08-23 audit round; an earlier draft of this comment stated
    9.44 MiB, off by ~95x). In production this branch is rarely taken at
    all: ``sc_iteration.py``'s own eigensolves (``distributed_eigh_bands``/
    ``eigh_kshard``) always hand ``rotate_wavefunctions`` an
    ALREADY-PLACED ``jax.Array`` U, which takes the no-op
    ``isinstance`` branch below and never reaches this replicated
    ``device_put`` at all — the host-numpy path this figure prices is a
    test/harness-only fallback. Still bounded and worth contrasting with
    the μ-class objects this layout exists to keep off any single rank
    (G/W/V at ``O(N_μ²/P)`` per rank, reaching multi-GB at production
    μ). This is the SAME bound legacy's own replicated-U path already
    accepted (see
    :func:`rotate_wavefunctions`'s docstring, "U IS NEVER REPLICATED"
    section — legacy's IS, in this one host-array corner case; only the
    SHARDED-U path that section describes avoids it, and this function's
    caller reshards immediately afterward inside
    :func:`_face_embed_active_U`).
    """
    if isinstance(U, jax.Array):
        return U
    from common.collectives import device_put_process_local
    U_np = np.asarray(U, dtype=np.complex128)
    return device_put_process_local(
        U_np, NamedSharding(mesh_xy, P(None, None, None)))


def _face_embed_active_U(U_active, *, nb_full: int, a_lo: int, mesh_xy: Mesh):
    """``(nk, nb_active, nb_active) -> (nk, nb_full, nb_full)``: identity
    outside ``[a_lo, a_lo+nb_active)``, ``U_active`` inside — i.e.
    ``blockdiag(U_active, I)`` at the active window's own offset.
    ``U_active`` must already be a placed ``jax.Array`` (see
    :func:`_place_U_face`, called by the caller before this — this
    function itself always runs traced, inside :func:`_face_rotate_kernel`
    's jit'd body).

    Mirrors the embedding idiom ``gw.qsgw_head._assemble_kernel`` already
    uses for the velocity/head manifold (``blockdiag(delta, 0)`` /
    ``blockdiag(U, I)``), for the SAME reason stated there, applied here
    to ψ instead of velocity: a face ψ copy's band axis is MESH-SHARDED
    (``PSI_NMU_SPEC`` on 'x', ``PSI_MUN_SPEC`` on 'y') and cannot be
    sliced to an arbitrary logical window — report obstacle #3,
    :meth:`Wavefunctions.band_mask`'s own "weight, don't window" doctrine.
    Embedding the identity into U instead of windowing ψ lets ONE GEMM,
    run over ψ's full ``[b0,b4)`` extent, reproduce legacy's
    slice-rotate-writeback EXACTLY: a rotation by a block-diagonal unitary
    is itself block-diagonal, so entries outside the active block are
    ``δ_mn`` (pass-through — "bands outside the window always keep their
    DFT ψ") and no active/inactive cross term is ever nonzero.

    Costs the FULL ``nb_full`` GEMM rather than a windowed one — the same
    named cost ``band_mask``'s own docstring states for its masked calls.
    """
    U_active = jnp.asarray(U_active, dtype=jnp.complex128)
    nk = int(U_active.shape[0])
    eye = jnp.broadcast_to(
        jnp.eye(nb_full, dtype=jnp.complex128)[None, :, :],
        (nk, nb_full, nb_full))
    U_full = jax.lax.dynamic_update_slice(eye, U_active, (0, a_lo, a_lo))
    return jax.lax.with_sharding_constraint(
        U_full, NamedSharding(mesh_xy, P(None, 'x', 'y')))


def _face_rotate_kernel(mesh: Mesh, a_lo: int, nb_active: int, nb_full: int,
                        n_rmu: int, ns: int, nk: int):
    """One built kernel per ``(mesh, active window, face shape)`` — mirrors
    :func:`_rotate_kernel`'s cache shape, but for the two-face carrier.

    Builds the TWO ``distrib_la.gemm_plan`` N,N GEMMs ONCE (their shapes —
    ``nb_full``/``n_rmu*ns``/``nk`` — are fixed for the whole run; only
    the embedding step varies per call, inside the same compiled program)
    and returns a jit'd ``fn(psi_nmu, psi_mun, U_active) -> (psi_nmu',
    psi_mun')``.

    THE TWO ROTATIONS, matching ``reports/gwjax_low_mem_bands_audit_2026-
    08-22/report.md``'s own census entry ("U^T @ psi_nmu, psi_mun @ U ...
    two face orientations of U"):

      psi_nmu' = U^T @ psi_nmu   (contract psi_nmu's OWN band axis, which
                 is already on 'x' — psi_nmu plugs in as gemm_plan's B
                 operand with NO reshard; U needs the "new band on x, old
                 band on y" orientation, which is the TRANSPOSE of U's
                 native ``band_rotation_spec`` — a genuine, but bounded
                 nb_full²-scale, resharding transpose, not a bitcast)
      psi_mun' = psi_mun @ U     (psi_mun plugs in as gemm_plan's A
                 operand with NO reshard; U plugs in as B DIRECTLY, at
                 its native ``band_rotation_spec`` orientation — FREE,
                 no reshard at all)

    So of the two orientations, only ONE (``psi_nmu``'s) needs the
    resharding transpose; the other is exactly what
    :func:`_face_embed_active_U` already returns.  Both merges use
    :func:`common.contract_bands.merge_spin_centroid`/
    :func:`split_spin_centroid` — the SAME (s,μ) GEMM-seam convention
    ``greens_function_kernel._face_build_G``/``contract_bands.
    _face_project_kernel`` already use; this function adds no third
    convention.

    Everything (embed, both merges, both planned GEMMs, both splits) runs
    inside ONE outer ``@jax.jit`` — the zeta-fit CCT port's own measured
    reason (``docs/architecture/zeta_fit_face_psi_cct.md``, "Folding BOTH
    gemm(...) calls ... into ONE outer @jax.jit fixed [an OOM] outright"):
    two separate top-level jit dispatches cannot share buffer assignment
    the way one fused program can.
    """
    from distrib_la import gemm_plan
    from common.contract_bands import merge_spin_centroid, split_spin_centroid

    key = (id(mesh), a_lo, nb_active, nb_full, n_rmu, ns, nk)
    hit = _FACE_ROTATE_CACHE.get(key)
    if hit is not None:
        return hit

    mu_s = n_rmu * ns
    plan_nmu = gemm_plan(mesh, m=nb_full, k=nb_full, n=mu_s, nq=nk,
                        dtype=jnp.complex128)
    plan_mun = gemm_plan(mesh, m=mu_s, k=nb_full, n=nb_full, nq=nk,
                        dtype=jnp.complex128)

    @jax.jit
    def fn(psi_nmu_full, psi_mun_full, U_active):
        U_full = _face_embed_active_U(
            U_active, nb_full=nb_full, a_lo=a_lo, mesh_xy=mesh)

        B_nmu = merge_spin_centroid(psi_nmu_full, 2, 3)   # (nk,nb_full,mu_s) P(_,'x','y')
        A_nmu = jax.lax.with_sharding_constraint(          # U^T: new-on-x, old-on-y
            jnp.swapaxes(U_full, 1, 2), NamedSharding(mesh, P(None, 'x', 'y')))
        D_nmu = plan_nmu(A_nmu, B_nmu)                     # (nk,nb_full,mu_s) P(_,'x','y')
        psi_nmu_out = split_spin_centroid(D_nmu, 2, ns, n_rmu)

        A_mun = merge_spin_centroid(psi_mun_full, 1, 2)    # (nk,mu_s,nb_full) P(_,'x','y')
        D_mun = plan_mun(A_mun, U_full)                    # (nk,mu_s,nb_full) P(_,'x','y')
        psi_mun_out = split_spin_centroid(D_mun, 1, ns, n_rmu)
        return psi_nmu_out, psi_mun_out

    _FACE_ROTATE_CACHE[key] = fn
    return fn


def _rotate_wavefunctions_legacy(
    wfns_dft: Wavefunctions,
    U_dft_to_qp_active: jax.Array,
    *,
    enk_active_new: jax.Array,
    enk_base: jax.Array | None = None,
    efermi: float | None,
    mesh_xy: Mesh,
    active_slice: slice | None = None,
) -> Wavefunctions:
    """The exact pre-``low_mem_bands`` body of ``rotate_wavefunctions``.
    UNTOUCHED (byte-for-byte, diff against a pre-face-rotation ref) — do
    not edit this function to add face-layout behaviour; add a sibling
    instead.  See :func:`rotate_wavefunctions` for the full docstring
    (COLUMN CONVENTION, U sharding, active/inactive partition) — kept on
    the public dispatcher rather than duplicated here, mirroring
    ``greens_function_kernel._legacy_build_G``'s own precedent.
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

    # A QP rotation returns only a numerical carrier.  The DFT binding held
    # by orchestration cannot follow it; a future QP receipt owner must issue
    # a new AuthenticatedWavefunctions object for this transformed source.
    return Wavefunctions(
        psi_xn=psi_xn, psi_xr=psi_xr, psi_yr=psi_yr, psi_yn=psi_yn,
        enk=enk_full, occ=occ_full, slices=wfns_dft.slices,
    )


def _rotate_wavefunctions_face(
    wfns_dft: Wavefunctions,
    U_dft_to_qp_active: jax.Array,
    *,
    enk_active_new: jax.Array,
    enk_base: jax.Array | None = None,
    efermi: float | None,
    mesh_xy: Mesh,
    active_slice: slice | None = None,
) -> Wavefunctions:
    """``layout='face'`` sibling of :func:`_rotate_wavefunctions_legacy` —
    same validation and the same ``enk``/``occ`` rebuild, but the ψ
    rotation itself routes through :func:`_face_rotate_kernel` (two
    planned N,N GEMMs against a block-embedded U) instead of
    :func:`_rotate_kernel` (slice + replicated-U contraction + writeback).
    See :func:`rotate_wavefunctions` for the shared docstring.
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

    with mesh_xy:
        fk = face_kernel_kwargs(wfns_dft)
        nk_face, nb_full, n_rmu, ns = fk['face_shape']
        U = _place_U_face(U_dft_to_qp_active, mesh_xy)
        rotate = _face_rotate_kernel(
            mesh_xy, a_lo, nb_active, nb_full, n_rmu, ns, nk_face)
        psi_nmu, psi_mun = rotate(wfns_dft.psi_nmu, wfns_dft.psi_mun, U)

        # enk/occ: SAME expression as the legacy sibling (layout-independent
        # — enk/occ are always P(None,None) replicated regardless of ψ
        # layout, see Wavefunctions.enk's own field comment).
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

    # As above, the host DFT binding cannot follow a QP-rotated carrier.
    return Wavefunctions(
        psi_nmu=psi_nmu, psi_mun=psi_mun,
        enk=enk_full, occ=occ_full, slices=wfns_dft.slices, layout='face',
    )


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

    Dispatches on ``wfns_dft.layout`` — no separate ``layout=`` parameter:
    the bundle already carries the tag, so it is the single owner of the
    choice, not each call site (``sc_iteration.py:1753`` needs no change
    to pick up the face path).  ``layout='legacy'`` routes to
    :func:`_rotate_wavefunctions_legacy`, the exact pre-``low_mem_bands``
    body.  ``layout='face'`` routes to :func:`_rotate_wavefunctions_face`
    — two planned ``distrib_la.gemm_plan`` N,N GEMMs
    (:func:`_face_rotate_kernel`) against a block-embedded U
    (:func:`_face_embed_active_U`) rather than a sliced-ψ contraction,
    because a face ψ copy's band axis is mesh-sharded and cannot be
    windowed (report census row "QSGW orbital rotation").

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
    if wfns_dft.layout == 'legacy':
        impl = _rotate_wavefunctions_legacy
    elif wfns_dft.layout == 'face':
        impl = _rotate_wavefunctions_face
    else:
        raise ValueError(
            f"rotate_wavefunctions: wfns_dft.layout={wfns_dft.layout!r} "
            f"not in ('legacy', 'face').")
    return impl(
        wfns_dft, U_dft_to_qp_active, enk_active_new=enk_active_new,
        enk_base=enk_base, efermi=efermi, mesh_xy=mesh_xy,
        active_slice=active_slice)


# ---------------------------------------------------------------------------
# Band-basis projection — Σ_mn(k) = Σ_{s,μ,s',μ'} ψ*_m(s,μ) Σ(s,μ,s',μ') ψ_n(s',μ')
#
# Lives here because the only state these contractions need is the (xr, yn)
# pair of sharded ψ copies that the bundle owns; consumers (cohsex_sigma,
# the AOT memory model) operate at the bundle's seam.
# ---------------------------------------------------------------------------

def project(psi_xr, psi_yn, sigma_k, *, layout='legacy', mesh_xy=None,
           face_shape=None, right_face_shape=None, face_project_fn=None):
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
            mesh_xy, layout='face', face_shape=face_shape,
            right_face_shape=right_face_shape)
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
