"""Store raw-parent wavefunctions and transient two-face children with their band metadata."""
from __future__ import annotations

import dataclasses
import functools
from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

from common.wfn_layout import PSI_MUN_SPEC, PSI_NMU_SPEC, psi_specs
from file_io.wfn_basis import WavefunctionBasisReceipt


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
# Canonical face specs are shared with the ISDF and band-contraction owners.
# ---------------------------------------------------------------------------

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
_LAYOUTS = ('face', 'axis')


@dataclass
class ParentGreenCarrier:
    """Raw-parent operands for Green contractions and band projections.

    ``psi_nmu``/``psi_mun`` are the raw WFN parent rows in the run's
    orbit-packed centroid order (the one in-memory order, so the plan's
    unfold is collective-free and every full-k operator -- Σ_k after the FFT
    convolution, V, W -- projects onto them directly).  ``n_parent`` k rows:
    the parent carrier holds no full-k wavefunction.
    """

    psi_nmu: jax.Array
    psi_mun: jax.Array
    enk: jax.Array
    occ: jax.Array
    plan: object
    layout: str = 'face'

    def __post_init__(self):
        """Validate the static layout independently of the spinor extent."""
        psi_specs(self.layout)

    def projection_faces(self):
        """Orient the existing copies along the operator's contracted centroid axes."""
        if self.layout == "axis":
            return (self.psi_mun.transpose(0, 3, 1, 2),
                    self.psi_nmu.transpose(0, 2, 3, 1))
        return self.psi_nmu, self.psi_mun

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
    meta_fields=['plan', 'layout'],
)


@dataclass
class Wavefunctions:
    """Keep raw-parent storage or transient child faces beside replicated band metadata."""

    enk: jax.Array       # (nk, nb_full) replicated
    #: Full-k occupation metadata; parent weights reside with the parent faces.
    occ: jax.Array
    slices: BandSlices
    psi_nmu: jax.Array | None = None  # (nk, n_X, s, μ_Y)
    psi_mun: jax.Array | None = None  # (nk, s, μ_X, n_Y)
    #: Raw-parent faces in the same packed centroid order as every operator.
    #: They are the only stored faces when all consumers support parents.
    green_parent: ParentGreenCarrier | None = None
    #: The canonical face layout is static pytree metadata.
    layout: str = "face"
    def __post_init__(self) -> None:
        if self.layout not in _LAYOUTS:
            raise ValueError(
                f"Wavefunctions: layout={self.layout!r} not in {_LAYOUTS}.")


    @functools.partial(jax.jit, static_argnames=('bands',))
    def band_mask(self, bands: slice) -> jax.Array:
        """Select the logical band interval without slicing a mesh-sharded face."""
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
    data_fields=['psi_nmu', 'psi_mun', 'green_parent', 'enk', 'occ'],
    meta_fields=['slices', 'layout'],
)


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
        if not isinstance(self.wavefunctions, Wavefunctions):
            raise TypeError(
                "AuthenticatedWavefunctions requires a Wavefunctions "
                f"carrier; got {type(self.wavefunctions).__name__}")
        if not isinstance(self.receipt, WavefunctionBasisReceipt):
            raise TypeError(
                "AuthenticatedWavefunctions requires a canonical basis "
                f"receipt; got {type(self.receipt).__name__}")
        self.receipt.assert_matches_carrier(
            self.wavefunctions, where="AuthenticatedWavefunctions")


def face_extents(wfns: "Wavefunctions") -> tuple[int, int, int]:
    """``(nk_full, nspinor, mu_padded)`` of a face bundle.

    Read off ``psi_mun`` when the full-k faces exist; when the run stores
    raw parents only (``gw_init`` parents-only storage: both faces
    ``None``), off the parent carrier -- its plan names the full-k row
    count and its faces carry the padded centroid extent every full-k
    operator is stored at.  Every full-k shape a kernel factory
    sizes itself by comes through here, so a parents-only bundle looks to
    the factories exactly like a full-k one.
    """
    if wfns.layout not in _LAYOUTS:
        raise ValueError(
            f"face_extents: layout={wfns.layout!r} carries no face.")
    if wfns.psi_mun is not None:
        nk, s, mu, _ = wfns.psi_mun.shape
        return int(nk), int(s), int(mu)
    carrier = wfns.green_parent
    if carrier is None:
        raise ValueError(
            "face_extents: this face bundle holds no full-k faces and no "
            "parent carrier; nothing names its shape.")
    return (int(carrier.plan.n_full), int(carrier.psi_mun.shape[1]),
            int(carrier.psi_mun.shape[2]))


def face_kernel_kwargs(wfns: "Wavefunctions", wfns_right=None) -> dict:
    """Read canonical endpoint face shapes from parent or transient child carriers."""
    if wfns_right is None:
        wfns_right = wfns
    if wfns.layout != wfns_right.layout:
        raise ValueError(
            "face_kernel_kwargs: endpoint layouts differ: "
            f"{wfns.layout!r} vs {wfns_right.layout!r}")
    nk, s, mu = face_extents(wfns)
    nk_r, s_r, mu_r = face_extents(wfns_right)
    left_shape = (nk, wfns.slices.nb_full, mu, s)
    right_shape = (nk_r, wfns_right.slices.nb_full, mu_r, s_r)
    result = {"layout": wfns.layout, "face_shape": left_shape}
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
        "layout": parent.layout,
        "face_shape": (nk, nb, nmu, ns),
        "k_unfold_plan": parent.plan,
    }


def sigma_face_kernel_kwargs(wfns: "Wavefunctions") -> dict:
    """:func:`face_kernel_kwargs` plus the carrier's ``k_unfold_plan``.

    ``face_shape`` names the full-k accumulator. The existing plan owns the
    parent count, parent rows and symmetry action used for both Green
    contraction and band projection. Factories retain the plan as an identity
    cache key; no per-call route copies its tables or shapes.
    """
    result = face_kernel_kwargs(wfns)
    carrier = wfns.green_parent
    if (not result or carrier is None
            or getattr(carrier.plan, 'parent_full_rows', None) is None
            or getattr(carrier.plan, 'sym', None) is None):
        return result
    result["k_unfold_plan"] = carrier.plan
    return result


def parent_sigma_operands(wfns: "Wavefunctions"):
    """The Σ kernel operands of the parent route, in the kernels' slot order.

    ``(psi_coh_xn, psi_coh_yr, psi_proj_xr, psi_proj_yn, enk, occ)``: the
    band-replicated parent faces for the G contraction (first operand direct, second
    conjugated inside ``build_G``), the same faces for the band projection
    (first operand conjugated inside the projector), and the parent-row
    energy/occupation tables.  Same six roles the full-k face call sites
    fill from ``wfns`` itself.
    """
    carrier = wfns.green_parent
    if carrier is None:
        raise ValueError(
            "parent_sigma_operands: the bundle carries no parent carrier.")
    proj_nmu, proj_mun = carrier.projection_faces()
    return (carrier.psi_mun, carrier.psi_nmu, proj_nmu, proj_mun,
            carrier.enk, carrier.occ)



def parent_faces(
    psi_rmu_Y_parent, psi_rmuT_X_parent, *, mesh_xy: Mesh, layout='face',
) -> tuple[jax.Array, jax.Array]:
    """Raw-parent loader outputs as the two face layouts.

    The same two constraints :func:`build_wavefunctions_face` applies to the
    full-k load, on ``n_parent`` rows.  The loader already sampled the run's
    packed centroid order, so nothing is permuted here.
    """
    nmu_spec, mun_spec = psi_specs(layout)
    with mesh_xy:
        psi_nmu = jax.lax.with_sharding_constraint(
            psi_rmu_Y_parent, NamedSharding(mesh_xy, nmu_spec))
        psi_mun = jax.lax.with_sharding_constraint(
            jnp.conj(psi_rmuT_X_parent).transpose(0, 3, 1, 2),
            NamedSharding(mesh_xy, mun_spec))
    return psi_nmu, psi_mun


def build_packed_parent_green_carrier(
    wfns: "Wavefunctions", psi_nmu_parent, psi_mun_parent, *, plan,
    mesh_xy: Mesh,
) -> ParentGreenCarrier:
    """Bind raw-parent faces to the full-k bundle's scalar tables."""
    nmu_spec, mun_spec = psi_specs(wfns.layout)
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
            psi_nmu_parent, NamedSharding(mesh_xy, nmu_spec))
        psi_mun = jax.lax.with_sharding_constraint(
            psi_mun_parent, NamedSharding(mesh_xy, mun_spec))
        enk = plan.parent_rows(wfns.enk)
        occ = plan.parent_rows(wfns.occ)
        rep2 = NamedSharding(mesh_xy, P(None, None))
        enk = jax.lax.with_sharding_constraint(enk, rep2)
        occ = jax.lax.with_sharding_constraint(occ, rep2)
    return ParentGreenCarrier(
        psi_nmu=psi_nmu, psi_mun=psi_mun, enk=enk, occ=occ, plan=plan,
        layout=wfns.layout)


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
    """Attach raw-parent loader outputs to a face bundle."""
    psi_nmu, psi_mun = parent_faces(
        psi_rmu_Y_parent, psi_rmuT_X_parent, mesh_xy=mesh_xy, layout=wfns.layout)
    return attach_packed_parent_green_carrier(
        wfns, psi_nmu, psi_mun, plan=plan, mesh_xy=mesh_xy)


def psi_field_names(layout: str) -> tuple[str, ...]:
    """Name the two canonical face orientations for residency accounting."""
    if layout != "face":
        raise ValueError(f"psi_field_names: unknown layout {layout!r}")
    return ("psi_nmu", "psi_mun")


def padded_centroid_extent(wfns: "Wavefunctions") -> int:
    """Read the packed centroid extent from the parent or transient face carrier."""
    return face_extents(wfns)[2]


def bundle_bytes_per_rank(wfns: "Wavefunctions") -> dict:
    """Count addressable bytes of stored parent and transient child faces on this rank."""
    out: dict = {}
    for f in psi_field_names(wfns.layout):
        arr = getattr(wfns, f)
        if arr is None:
            continue
        out[f] = int(sum(int(s.data.nbytes) for s in arr.addressable_shards))
    if wfns.green_parent is not None:
        for f in ("psi_nmu", "psi_mun"):
            arr = getattr(wfns.green_parent, f)
            if arr is None:
                continue
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


def build_wavefunctions_face(
    psi_rmu_Y, psi_rmuT_X, *, enk_full, slices, mesh_xy, efermi=None,
    basis_receipt=None, layout="face",
) -> Wavefunctions:
    """Assemble canonical faces from direct Y samples and conjugated X samples."""
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
        enk=enk_full, occ=occ_full, slices=slices, layout=layout)
    if basis_receipt is not None:
        basis_receipt.assert_matches_carrier(
            wfns, where="build_wavefunctions_face")
    return wfns


def wavefunctions_face_from_restart(
    psi_nmu, psi_mun, *, enk_full, slices, mesh_xy, efermi=None,
    basis_receipt=None, layout="face",
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
        enk=enk_full, occ=occ_full, slices=slices, layout=layout)
    if basis_receipt is not None:
        basis_receipt.assert_matches_carrier(
            wfns, where="wavefunctions_face_from_restart")
    return wfns


# ---------------------------------------------------------------------------
# Self-consistent QSGW: rotate the bundle into a new band basis
# ---------------------------------------------------------------------------

_FACE_ROTATE_CACHE: dict = {}


def _place_U_face(U, mesh_xy: Mesh):
    """Place the band rotation on its declared mesh without changing a host or device source convention."""
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
    """One built kernel per ``(mesh, active window, face shape)``.

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
    """Rotate the active band columns, preserve inactive wavefunctions, and rebuild energies and occupations."""
    if wfns_dft.layout != "face":
        raise ValueError("rotate_wavefunctions requires the two-face carrier.")
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
        psi_nmu = psi_mun = None
        carrier_rotated = None
        if wfns_dft.psi_nmu is not None:
            rotate = _face_rotate_kernel(
                mesh_xy, a_lo, nb_active, nb_full, n_rmu, ns, nk_face)
            psi_nmu, psi_mun = rotate(wfns_dft.psi_nmu, wfns_dft.psi_mun, U)
            if wfns_dft.green_parent is not None:
                # A carrier beside full-k faces (the self-consistent map
                # keeps both) rotates with them, so every iteration's
                # screening and Sigma take the same route as iteration 0.
                carrier_rotated = _rotate_parent_carrier(
                    wfns_dft.green_parent, U, a_lo=a_lo, nb_active=nb_active,
                    nb_full=nb_full, ns=ns, mesh_xy=mesh_xy)
        else:
            # Parents-only storage: the carrier is the run's only ψ.  Rotate
            # its faces with U on the parents' OWN full-k rows -- the
            # transported child basis is the parent basis, so the map's
            # rotation at a child row is the parent's (conjugated on
            # antiunitary rows, which the carrier never materializes).
            carrier_rotated = _rotate_parent_carrier(
                wfns_dft.green_parent, U, a_lo=a_lo, nb_active=nb_active,
                nb_full=nb_full, ns=ns, mesh_xy=mesh_xy)

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
    rotated = Wavefunctions(
        psi_nmu=psi_nmu, psi_mun=psi_mun,
        enk=enk_full, occ=occ_full, slices=wfns_dft.slices, layout='face',
    )
    if carrier_rotated is None:
        return rotated
    packed_nmu, packed_mun, plan = carrier_rotated
    return dataclasses.replace(
        rotated,
        green_parent=build_packed_parent_green_carrier(
            rotated, packed_nmu, packed_mun, plan=plan, mesh_xy=mesh_xy))


def _rotate_parent_carrier(carrier, U_placed, *, a_lo, nb_active, nb_full,
                           ns, mesh_xy):
    """Rotate each raw parent with the band matrix at its authenticated full-k row."""
    if carrier is None:
        raise ValueError(
            "rotate_wavefunctions: a face bundle without full-k faces needs "
            "a parent carrier to rotate.")
    plan = carrier.plan
    rows = getattr(plan, 'parent_full_rows', None)
    if rows is None:
        raise ValueError(
            "rotate_wavefunctions: the parent plan names no full-k rows "
            "(hand-assembled plan); cannot select the parents' rotation.")
    U_par = jax.lax.with_sharding_constraint(
        jnp.take(U_placed, jnp.asarray(np.asarray(rows), dtype=jnp.int32),
                 axis=0),
        NamedSharding(mesh_xy, P(None, None, None)))
    n_parent = int(plan.n_parent)
    rot = _face_rotate_kernel(
        mesh_xy, a_lo, nb_active, nb_full, int(plan.n_centroid_packed), ns,
        n_parent)
    packed_nmu, packed_mun = rot(carrier.psi_nmu, carrier.psi_mun, U_par)
    return packed_nmu, packed_mun, plan


# ---------------------------------------------------------------------------
# Band-basis projection — Σ_mn(k) = Σ_{s,μ,s',μ'} ψ*_m(s,μ) Σ(s,μ,s',μ') ψ_n(s',μ')
#
# Lives here because the only state these contractions need is the (xr, yn)
# pair of sharded ψ copies that the bundle owns; consumers (cohsex_sigma,
# the AOT memory model) operate at the bundle's seam.
# ---------------------------------------------------------------------------

def project(psi_xr, psi_yn, sigma_k, *, layout='face', mesh_xy=None,
           face_shape=None, right_face_shape=None, face_project_fn=None):
    """Project a centroid operator through the shared canonical face contraction."""
    if layout not in _LAYOUTS:
        raise ValueError(
            f"project: layout must be 'face', got {layout!r}")
    fn = face_project_fn
    if fn is None:
        if mesh_xy is None or face_shape is None:
            raise ValueError(
                "project(layout='face') requires either face_project_fn=, "
                "or both mesh_xy= and face_shape= to build one inline "
                "(see common.contract_bands.contract_bands_block_reshard)")
        from common.contract_bands import contract_bands_block_reshard
        fn = contract_bands_block_reshard(
            mesh_xy, layout=layout, face_shape=face_shape,
            right_face_shape=right_face_shape)
    return fn(psi_xr, sigma_k, psi_yn)
