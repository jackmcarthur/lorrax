"""Canonical wavefunction storage for ISDF-basis GW calculations.

Provides four device-distributed copies of ψ_nk(r_μ) optimised for the two
fundamental ISDF operations:

  Construction (G, χ₀):  contract over band index → output (μ_X, μ_Y)
  Projection   (Σ→bands): contract over centroid index → output (m, n)

Each copy differs in two independent choices:
  • Device axis: x or y — which mesh axis the centroid dimension μ is sharded across.
  • Memory layout: n-fast (bands last) or r-fast (centroids last).

The combination determines which einsum contractions are cache-friendly and
communication-free on the 2-D device mesh.

Naming convention for the four copies:  psi_{device}_{fast_axis}

  psi_xn : (nk, s, μ_X, n)   bands fast, μ on X   →  G/χ₀ construction LHS (conj)
  psi_xr : (nk, n, s, μ_X)   centroids fast, μ on X   →  Σ projection LHS (conj)
  psi_yr : (nk, n, s, μ_Y)   centroids fast, μ on Y   →  G/χ₀ construction RHS
  psi_yn : (nk, s, μ_Y, n)   bands fast, μ on Y   →  Σ projection RHS

In all four layouts the spinor index s sits adjacent to the centroid index μ,
so contractions that sum over (s, μ) pairs sweep contiguous memory.
"""
from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace

import jax
import jax.numpy as jnp
from jax.sharding import Mesh, PartitionSpec as P


# ---------------------------------------------------------------------------
# Band-edge bookkeeping
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class BandSlices:
    """Precomputed local slices derived from the five canonical band edges.

    Band edges (global indices):
        b0  lowest band in the calculation
        b1  start of mixed valence/conduction sigma region
        b2  LUMO (first unoccupied, = number of occupied)
        b3  end of the sigma/QP evaluation window
        b4  highest band (end of full computational window)

    All slices are LOCAL (relative to b0, so valence starts at 0).
    """

    b0: int
    b1: int
    b2: int
    b3: int
    b4: int

    # Primary slices — these are the only ones used in active code.
    val:   slice   # [0, b2-b0)         valence bands
    cond:  slice   # [b2-b0, b4-b0)     conduction bands
    sigma: slice   # [0, b3-b0)         QP evaluation window (occ + sigma cond)
    full:  slice   # [0, b4-b0)         all bands
    occ:   slice   # [0, b2-b0)         occupied (same as val for insulators)

    @classmethod
    def from_band_edges(
        cls, b0: int, b1: int, b2: int, b3: int, b4: int
    ) -> BandSlices:
        if not (b0 <= b1 <= b2 <= b3 <= b4):
            raise ValueError(
                f"Invalid band edges: expected b0<=b1<=b2<=b3<=b4, "
                f"got {(b0, b1, b2, b3, b4)}"
            )
        nb_full = b4 - b0
        return cls(
            b0=b0, b1=b1, b2=b2, b3=b3, b4=b4,
            val=slice(0, b2 - b0),
            cond=slice(b2 - b0, nb_full),
            sigma=slice(0, b3 - b0),
            full=slice(0, nb_full),
            occ=slice(0, b2 - b0),
        )

    @property
    def nb_full(self) -> int:
        return self.b4 - self.b0

    @property
    def nb_sigma(self) -> int:
        return self.b3 - self.b0

    # ---- Backward-compatibility aliases (to be removed once all callers
    #      are migrated) ----
    @property
    def l_slice(self) -> slice:
        return self.sigma

    @property
    def coh_slice(self) -> slice:
        return self.full

    @property
    def v_slice(self) -> slice:
        return self.val

    @property
    def c_slice(self) -> slice:
        return self.cond

    @property
    def occ_slice(self) -> slice:
        return self.occ

    @property
    def full_slice(self) -> slice:
        return self.full

    @property
    def r_slice(self) -> slice:
        return slice(self.b1 - self.b0, self.b4 - self.b0)

    @property
    def sigma_slice(self) -> slice:
        return slice(self.b1 - self.b0, self.b3 - self.b0)

    @property
    def nb_l(self) -> int:
        return self.nb_sigma


# ---------------------------------------------------------------------------
# Wavefunction storage
# ---------------------------------------------------------------------------

# Sharding specs for the four copies (assuming 2-D mesh with axes 'x', 'y').
# The centroid dimension μ is the one being sharded.
PSI_XN_SPEC = P(None, None, 'x', None)   # (nk, s, μ_X, n)
PSI_XR_SPEC = P(None, None, None, 'x')   # (nk, n, s, μ_X)
PSI_YR_SPEC = P(None, None, None, 'y')   # (nk, n, s, μ_Y)
PSI_YN_SPEC = P(None, None, 'y', None)   # (nk, s, μ_Y, n)


@dataclass
class Wavefunctions:
    """Four device-distributed copies of ψ_nk(r_μ) for ISDF-basis GW.

    All copies span the full band range [b0, b4).  Use the slice accessors
    ``xn``, ``xr``, ``yr``, ``yn`` to select sub-ranges; they hide which
    axis the band index lives on in each layout.

    Attributes
    ----------
    psi_xn : jax.Array, shape (nk, s, μ, nb_full)
        Bands fast (last axis), μ sharded on device axis X.
        Used as the **conjugate** operand in G / χ₀ construction.
    psi_xr : jax.Array, shape (nk, nb_full, s, μ)
        Centroids fast (last axis), μ sharded on device axis X.
        Used as the **conjugate** operand in Σ → band projection.
    psi_yr : jax.Array, shape (nk, nb_full, s, μ)
        Centroids fast (last axis), μ sharded on device axis Y.
        Used as the **direct** operand in G / χ₀ construction.
    psi_yn : jax.Array, shape (nk, s, μ, nb_full)
        Bands fast (last axis), μ sharded on device axis Y.
        Used as the **direct** operand in Σ → band projection.
    enk : jax.Array, shape (nk, nb_full)
        Band energies (Rydberg), replicated.
    occ : jax.Array, shape (nk, nb_full)
        Occupation numbers (1.0 occupied, 0.0 empty), replicated.
    slices : BandSlices
        Canonical band-edge bookkeeping and precomputed slices.
    """

    psi_xn: jax.Array   # (nk, s, μ_X, n)   bands fast, μ on X
    psi_xr: jax.Array   # (nk, n, s, μ_X)   centroids fast, μ on X
    psi_yr: jax.Array   # (nk, n, s, μ_Y)   centroids fast, μ on Y
    psi_yn: jax.Array   # (nk, s, μ_Y, n)   bands fast, μ on Y

    enk: jax.Array       # (nk, nb_full) replicated
    occ: jax.Array       # (nk, nb_full) replicated
    slices: BandSlices

    # ---- Band-slicing accessors ----
    # These hide which axis carries the band index in each layout.

    def xn(self, bands: slice) -> jax.Array:
        """X-sharded, bands fast — sliced along the band axis (last)."""
        return self.psi_xn[:, :, :, bands]

    def xr(self, bands: slice) -> jax.Array:
        """X-sharded, centroids fast — sliced along the band axis (axis 1)."""
        return self.psi_xr[:, bands, :, :]

    def yr(self, bands: slice) -> jax.Array:
        """Y-sharded, centroids fast — sliced along the band axis (axis 1)."""
        return self.psi_yr[:, bands, :, :]

    def yn(self, bands: slice) -> jax.Array:
        """Y-sharded, bands fast — sliced along the band axis (last)."""
        return self.psi_yn[:, :, :, bands]

    # ---- Backward-compatibility shims (delegate to the appropriate copy) ----

    @property
    def psi_x(self) -> jax.Array:
        """Legacy alias: X-sharded, bands-fast = psi_xn."""
        return self.psi_xn

    @property
    def psi_y(self) -> jax.Array:
        """Legacy alias: Y-sharded, centroids-fast = psi_yr."""
        return self.psi_yr

    def x(self, band_slice: slice) -> jax.Array:
        """Legacy alias for xn()."""
        return self.xn(band_slice)

    def y(self, band_slice: slice) -> jax.Array:
        """Legacy alias for yr()."""
        return self.yr(band_slice)


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------

def _assemble_full_from_left_right(
    psi_l_yr: jax.Array,
    psi_r_yr: jax.Array | None,
    slices: BandSlices,
) -> jax.Array:
    """Build canonical b0:b4 array in yr layout from legacy left/right arrays.

    Both inputs are expected in yr layout: (nk, nb, s, μ).
    """
    nb_full = slices.nb_full
    if psi_l_yr.shape[1] >= nb_full:
        return psi_l_yr[:, :nb_full, :, :]

    if psi_r_yr is None:
        raise ValueError(
            "Cannot assemble full b0:b4 wavefunctions: psi_l does not "
            "contain full range and psi_r is missing."
        )

    nb_v = slices.b2 - slices.b0
    nb_c = slices.b4 - slices.b2
    r_c_start = slices.b2 - slices.b1

    if psi_l_yr.shape[1] < nb_v:
        raise ValueError(
            f"psi_l has {psi_l_yr.shape[1]} bands but needs at least "
            f"{nb_v} for valence slice."
        )
    if psi_r_yr.shape[1] < (r_c_start + nb_c):
        raise ValueError(
            f"psi_r has {psi_r_yr.shape[1]} bands but needs at least "
            f"{r_c_start + nb_c} to form conduction slice."
        )

    psi_v = psi_l_yr[:, :nb_v, :, :]
    psi_c = psi_r_yr[:, r_c_start:(r_c_start + nb_c), :, :]
    return jnp.concatenate([psi_v, psi_c], axis=1)


def _build_four_copies(
    psi_yr_full: jax.Array,
    mesh_xy: Mesh,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    """From a single yr-layout array, build all four sharded copies.

    Input: psi_yr_full with shape (nk, nb_full, s, μ).
    Returns: (psi_xn, psi_xr, psi_yr, psi_yn).
    """
    from jax.sharding import NamedSharding

    with mesh_xy:
        # yr: (nk, n, s, μ_Y) — centroids fast, μ on Y
        psi_yr = jax.lax.with_sharding_constraint(
            psi_yr_full, NamedSharding(mesh_xy, PSI_YR_SPEC))

        # xn: (nk, s, μ_X, n) — transpose yr then reshard to X
        psi_xn = jax.lax.with_sharding_constraint(
            psi_yr_full.transpose(0, 2, 3, 1),
            NamedSharding(mesh_xy, PSI_XN_SPEC))

        # xr: (nk, n, s, μ_X) — same shape as yr, resharded to X
        psi_xr = jax.lax.with_sharding_constraint(
            psi_yr_full, NamedSharding(mesh_xy, PSI_XR_SPEC))

        # yn: (nk, s, μ_Y, n) — transpose yr, keep on Y
        psi_yn = jax.lax.with_sharding_constraint(
            psi_yr_full.transpose(0, 2, 3, 1),
            NamedSharding(mesh_xy, PSI_YN_SPEC))

    return psi_xn, psi_xr, psi_yr, psi_yn


def build_wavefunctions(
    psi_l_yr: jax.Array,
    psi_r_yr: jax.Array | None,
    *,
    enk_full: jax.Array,
    slices: BandSlices,
    mesh_xy: Mesh,
    efermi: float | None = None,
) -> Wavefunctions:
    """Construct all four wavefunction copies from left/right arrays.

    Parameters
    ----------
    psi_l_yr : array, shape (nk, nb_l, s, μ)
        Left (valence + sigma-window conduction) wavefunctions at ISDF
        centroids, in yr layout.
    psi_r_yr : array or None, shape (nk, nb_r, s, μ)
        Right (sigma-conduction + high conduction) wavefunctions.  Only
        needed when psi_l_yr does not already span the full band range.
    enk_full : array, shape (nk, nb_full)
        Band energies spanning [b0, b4), in Rydberg.
    slices : BandSlices
        Band-edge bookkeeping.
    mesh_xy : jax.sharding.Mesh
        2-D device mesh with axes ('x', 'y').
    efermi : float or None
        If given, occupations are derived from enk <= efermi.  Otherwise
        a step function at the b2 boundary is used.
    """
    from jax.sharding import NamedSharding

    psi_yr_full = _assemble_full_from_left_right(psi_l_yr, psi_r_yr, slices)
    if psi_yr_full.shape[1] != slices.nb_full:
        raise ValueError(
            f"Expected full band span {slices.nb_full}, "
            f"got {psi_yr_full.shape[1]}"
        )
    if enk_full.shape[1] != slices.nb_full:
        raise ValueError(
            f"enk_full must span b0:b4 ({slices.nb_full} bands), "
            f"got {enk_full.shape[1]}"
        )

    if efermi is None:
        occ_full = jnp.zeros_like(enk_full, dtype=jnp.float64)
        occ_full = occ_full.at[:, slices.occ].set(1.0)
    else:
        occ_full = jnp.asarray(enk_full <= efermi, dtype=jnp.float64)

    psi_xn, psi_xr, psi_yr, psi_yn = _build_four_copies(psi_yr_full, mesh_xy)

    rep2 = NamedSharding(mesh_xy, P(None, None))
    with mesh_xy:
        enk_full = jax.lax.with_sharding_constraint(enk_full, rep2)
        occ_full = jax.lax.with_sharding_constraint(occ_full, rep2)

    return Wavefunctions(
        psi_xn=psi_xn,
        psi_xr=psi_xr,
        psi_yr=psi_yr,
        psi_yn=psi_yn,
        enk=enk_full,
        occ=occ_full,
        slices=slices,
    )


def build_wavefunctions_from_full(
    psi_yr_full: jax.Array,
    *,
    enk_full: jax.Array,
    slices: BandSlices,
    mesh_xy: Mesh,
    efermi: float | None = None,
    psi_xn_full: jax.Array | None = None,
) -> Wavefunctions:
    """Construct all four copies from a full-band yr-layout array.

    If ``psi_xn_full`` is provided it is used directly (avoids one
    transpose+reshard); otherwise it is derived from ``psi_yr_full``.
    """
    from jax.sharding import NamedSharding

    if psi_yr_full.shape[1] != slices.nb_full:
        raise ValueError(
            f"Expected psi_yr_full to span b0:b4 ({slices.nb_full} bands), "
            f"got {psi_yr_full.shape[1]}"
        )
    if enk_full.shape[1] != slices.nb_full:
        raise ValueError(
            f"Expected enk_full to span b0:b4 ({slices.nb_full} bands), "
            f"got {enk_full.shape[1]}"
        )

    if efermi is None:
        occ_full = jnp.zeros_like(enk_full, dtype=jnp.float64)
        occ_full = occ_full.at[:, slices.occ].set(1.0)
    else:
        occ_full = jnp.asarray(enk_full <= efermi, dtype=jnp.float64)

    if psi_xn_full is not None:
        # Caller provided the xn copy already — build the other three.
        with mesh_xy:
            psi_yr = jax.lax.with_sharding_constraint(
                psi_yr_full, NamedSharding(mesh_xy, PSI_YR_SPEC))
            psi_xn = jax.lax.with_sharding_constraint(
                psi_xn_full, NamedSharding(mesh_xy, PSI_XN_SPEC))
            psi_xr = jax.lax.with_sharding_constraint(
                psi_yr_full, NamedSharding(mesh_xy, PSI_XR_SPEC))
            psi_yn = jax.lax.with_sharding_constraint(
                psi_yr_full.transpose(0, 2, 3, 1),
                NamedSharding(mesh_xy, PSI_YN_SPEC))
    else:
        psi_xn, psi_xr, psi_yr, psi_yn = _build_four_copies(
            psi_yr_full, mesh_xy)

    rep2 = NamedSharding(mesh_xy, P(None, None))
    with mesh_xy:
        enk_full = jax.lax.with_sharding_constraint(enk_full, rep2)
        occ_full = jax.lax.with_sharding_constraint(occ_full, rep2)

    return Wavefunctions(
        psi_xn=psi_xn,
        psi_xr=psi_xr,
        psi_yr=psi_yr,
        psi_yn=psi_yn,
        enk=enk_full,
        occ=occ_full,
        slices=slices,
    )


# ---------------------------------------------------------------------------
# Backward-compatibility aliases (to be removed after full migration)
# ---------------------------------------------------------------------------

WavefunctionBundle = Wavefunctions

build_wavefunction_bundle = build_wavefunctions
build_wavefunction_bundle_from_full = build_wavefunctions_from_full


def build_sigma_views(
    wf_bundle: Wavefunctions,
    mesh_xy: Mesh,
    sh: SimpleNamespace,
) -> SimpleNamespace:
    """Build legacy sigma-kernel wavefunction views.

    DEPRECATED — callers should use wf_bundle.xn/xr/yr/yn directly.
    Retained only until all call sites are migrated.
    """
    s = wf_bundle.slices
    return SimpleNamespace(
        psi_lT=wf_bundle.xn(s.sigma),
        psi_l=wf_bundle.yr(s.sigma),
        psi_cohT=wf_bundle.xn(s.full),
        psi_coh=wf_bundle.yr(s.full),
        psi_proj=wf_bundle.xr(s.sigma),
        psi_projT=wf_bundle.yn(s.sigma),
    )
