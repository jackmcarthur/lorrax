"""Canonical wavefunction storage for ISDF-basis GW calculations.

Four device-distributed copies of ψ_nk(r_μ), one for each combination of
{device axis} × {memory layout}:

  psi_xn : (nk, s, μ_X, n)  bands fast, μ on X  →  G/χ₀ LHS (conj)
  psi_xr : (nk, n, s, μ_X)  centroids fast, μ on X  →  Σ projection LHS (conj)
  psi_yr : (nk, n, s, μ_Y)  centroids fast, μ on Y  →  G/χ₀ RHS
  psi_yn : (nk, s, μ_Y, n)  bands fast, μ on Y  →  Σ projection RHS

In all four layouts the spinor index s sits adjacent to the centroid
index μ, so contractions that sum over (s, μ) pairs sweep contiguous memory.
"""
from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P


# ---------------------------------------------------------------------------
# Band-edge bookkeeping
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class BandSlices:
    """Precomputed local slices from the five canonical band edges.

    Band edges (global indices):
        b0  lowest band in the calculation
        b1  start of mixed valence/conduction sigma region
        b2  LUMO (first unoccupied)
        b3  end of the sigma/QP evaluation window
        b4  highest band (end of full computational window)

    All slices are LOCAL (relative to b0).
    """
    b0: int
    b1: int
    b2: int
    b3: int
    b4: int
    val:   slice   # [0, b2-b0)       valence
    cond:  slice   # [b2-b0, b4-b0)   conduction
    sigma: slice   # [0, b3-b0)       QP evaluation window
    full:  slice   # [0, b4-b0)       all bands
    occ:   slice   # [0, b2-b0)       occupied

    @classmethod
    def from_band_edges(cls, b0: int, b1: int, b2: int, b3: int, b4: int) -> BandSlices:
        if not (b0 <= b1 <= b2 <= b3 <= b4):
            raise ValueError(f"Invalid band edges: {(b0, b1, b2, b3, b4)}")
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

    @property
    def sigma_range(self) -> tuple[int, int]:
        """Global (start, end) for sigma band window: (b0, b3)."""
        return (self.b0, self.b3)

    @property
    def full_range(self) -> tuple[int, int]:
        """Global (start, end) for full band window: (b0, b4)."""
        return (self.b0, self.b4)


# ---------------------------------------------------------------------------
# Sharding specs for the four copies (2-D mesh with axes 'x', 'y').
# ---------------------------------------------------------------------------
PSI_XN_SPEC = P(None, None, 'x', None)   # (nk, s, μ_X, n)
PSI_XR_SPEC = P(None, None, None, 'x')   # (nk, n, s, μ_X)
PSI_YR_SPEC = P(None, None, None, 'y')   # (nk, n, s, μ_Y)
PSI_YN_SPEC = P(None, None, 'y', None)   # (nk, s, μ_Y, n)


# ---------------------------------------------------------------------------
# Wavefunction storage
# ---------------------------------------------------------------------------

@dataclass
class Wavefunctions:
    """Four device-distributed copies of ψ_nk(r_μ) spanning [b0, b4)."""

    psi_xn: jax.Array   # (nk, s, μ_X, n)
    psi_xr: jax.Array   # (nk, n, s, μ_X)
    psi_yr: jax.Array   # (nk, n, s, μ_Y)
    psi_yn: jax.Array   # (nk, s, μ_Y, n)
    enk: jax.Array       # (nk, nb_full) replicated
    occ: jax.Array       # (nk, nb_full) replicated
    slices: BandSlices

    def xn(self, bands: slice) -> jax.Array:
        return self.psi_xn[:, :, :, bands]

    def xr(self, bands: slice) -> jax.Array:
        return self.psi_xr[:, bands, :, :]

    def yr(self, bands: slice) -> jax.Array:
        return self.psi_yr[:, bands, :, :]

    def yn(self, bands: slice) -> jax.Array:
        return self.psi_yn[:, :, :, bands]


# Register as JAX pytree so Wavefunctions can be passed to @jax.jit functions.
# Array fields are traced; slices (static metadata) are compile-time constants.
jax.tree_util.register_dataclass(
    Wavefunctions,
    data_fields=['psi_xn', 'psi_xr', 'psi_yr', 'psi_yn', 'enk', 'occ'],
    meta_fields=['slices'],
)


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------

def _assemble_full_yr(psi_l_yr, psi_r_yr, slices):
    """Build full b0:b4 array in yr layout from left/right halves."""
    nb_full = slices.nb_full
    if psi_l_yr.shape[1] >= nb_full:
        return psi_l_yr[:, :nb_full, :, :]
    if psi_r_yr is None:
        raise ValueError("psi_l does not span full range and psi_r is missing.")
    nb_v = slices.b2 - slices.b0
    nb_c = slices.b4 - slices.b2
    r_c_start = slices.b2 - slices.b1
    psi_v = psi_l_yr[:, :nb_v, :, :]
    psi_c = psi_r_yr[:, r_c_start:(r_c_start + nb_c), :, :]
    return jnp.concatenate([psi_v, psi_c], axis=1)


def _build_four_copies(psi_yr_full, mesh_xy):
    """From a single yr-layout array, build all four sharded copies."""
    with mesh_xy:
        psi_yr = jax.lax.with_sharding_constraint(
            psi_yr_full, NamedSharding(mesh_xy, PSI_YR_SPEC))
        psi_xn = jax.lax.with_sharding_constraint(
            psi_yr_full.transpose(0, 2, 3, 1),
            NamedSharding(mesh_xy, PSI_XN_SPEC))
        psi_xr = jax.lax.with_sharding_constraint(
            psi_yr_full, NamedSharding(mesh_xy, PSI_XR_SPEC))
        psi_yn = jax.lax.with_sharding_constraint(
            psi_yr_full.transpose(0, 2, 3, 1),
            NamedSharding(mesh_xy, PSI_YN_SPEC))
    return psi_xn, psi_xr, psi_yr, psi_yn


def _build_occ(enk_full, slices, efermi):
    """Build occupation array (nk, nb_full), float64 in {0.0, 1.0}.

    Host-side: enk_full is tiny (nk × nb, usually < 10K doubles) so the
    D2H cost is trivial compared to the 8 pjit compiles the all-jnp
    version emitted at trace time.  Returns a jax.Array; the caller
    applies sharding constraints.
    """
    enk_host = np.asarray(enk_full)
    if efermi is None:
        occ = np.zeros_like(enk_host, dtype=np.float64)
        occ[:, slices.occ] = 1.0
    else:
        occ = (enk_host <= float(efermi)).astype(np.float64)
    return jnp.asarray(occ)


def build_wavefunctions(
    psi_l_yr, psi_r_yr, *, enk_full, slices, mesh_xy, efermi=None,
) -> Wavefunctions:
    """Build all four copies from left/right yr-layout arrays."""
    psi_yr_full = _assemble_full_yr(psi_l_yr, psi_r_yr, slices)
    occ_full = _build_occ(enk_full, slices, efermi)
    psi_xn, psi_xr, psi_yr, psi_yn = _build_four_copies(psi_yr_full, mesh_xy)
    rep2 = NamedSharding(mesh_xy, P(None, None))
    with mesh_xy:
        enk_full = jax.lax.with_sharding_constraint(enk_full, rep2)
        occ_full = jax.lax.with_sharding_constraint(occ_full, rep2)
    return Wavefunctions(psi_xn=psi_xn, psi_xr=psi_xr, psi_yr=psi_yr,
                         psi_yn=psi_yn, enk=enk_full, occ=occ_full, slices=slices)


def build_wavefunctions_from_full(
    psi_yr_full, *, enk_full, slices, mesh_xy, efermi=None, psi_xn_full=None,
) -> Wavefunctions:
    """Build all four copies from a full-band yr-layout array."""
    occ_full = _build_occ(enk_full, slices, efermi)
    if psi_xn_full is not None:
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
        psi_xn, psi_xr, psi_yr, psi_yn = _build_four_copies(psi_yr_full, mesh_xy)
    rep2 = NamedSharding(mesh_xy, P(None, None))
    with mesh_xy:
        enk_full = jax.lax.with_sharding_constraint(enk_full, rep2)
        occ_full = jax.lax.with_sharding_constraint(occ_full, rep2)
    return Wavefunctions(psi_xn=psi_xn, psi_xr=psi_xr, psi_yr=psi_yr,
                         psi_yn=psi_yn, enk=enk_full, occ=occ_full, slices=slices)
