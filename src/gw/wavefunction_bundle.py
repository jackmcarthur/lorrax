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
    """Concatenate left/right halves along the band axis — yr layout.

    psi_{l,r}_yr shape: (nk, nb, ns, n_rmu).  Band axis is axis 1.
    Axis 1 is replicated under PSI_YR_SPEC, so the concat is local.
    """
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


def _assemble_full_xn(psi_l_xn, psi_r_xn, slices):
    """Concatenate left/right halves along the band axis — xn layout.

    The ISDF zeta kernel stores x-sharded psi as ``psi_rmuT = conj(ψ)``
    so the conjugated factor is prefilled for its pair-density kernel.
    Downstream consumers (chi0 build_G, Σ_kij) expect psi_xn to be the
    *un-conjugated* ψ (they conjugate the yr side themselves).  We
    take ``jnp.conj`` here to recover that convention.

    Zeta layout: ``(nk, n_rmu, nb, ns)`` — μ on axis 1 (x-sharded),
    bands on axis 2 (replicated).  :func:`_build_four_copies` does the
    final transpose into PSI_XN_SPEC without crossing device boundaries.
    """
    nb_full = slices.nb_full
    if psi_l_xn.shape[2] >= nb_full:
        return jnp.conj(psi_l_xn[:, :, :nb_full, :])
    if psi_r_xn is None:
        raise ValueError("psi_l_xn does not span full range and psi_r_xn is missing.")
    nb_v = slices.b2 - slices.b0
    nb_c = slices.b4 - slices.b2
    r_c_start = slices.b2 - slices.b1
    psi_v = psi_l_xn[:, :, :nb_v, :]
    psi_c = psi_r_xn[:, :, r_c_start:(r_c_start + nb_c), :]
    return jnp.conj(jnp.concatenate([psi_v, psi_c], axis=2))


def _build_four_copies(psi_yr_full, mesh_xy, *, psi_xn_full=None):
    """Build the four sharded psi views chi0 / Σ consume.

    Layouts (see PSI_*_SPEC at the top of the module):
      psi_yr   (nk, nb,  ns,  μ_Y)     μ sharded on y
      psi_yn   (nk, ns,  μ_Y, nb)      μ sharded on y
      psi_xn   (nk, ns,  μ_X, nb)      μ sharded on x
      psi_xr   (nk, nb,  ns,  μ_X)     μ sharded on x

    The y-sharded pair is always cheap to derive from ``psi_yr_full``
    (transpose is a metadata op; sharding already matches).

    The x-sharded pair needs μ moved from y to x.  If the caller hands
    in ``psi_xn_full`` (as the ISDF zeta kernel already produces, both
    layouts as a byproduct of the fit), we derive psi_xn / psi_xr from
    it — no cross-device movement.  Otherwise we fall back to resharding
    ``psi_yr_full``, which costs two y→x all-to-alls on the μ axis.
    """
    with mesh_xy:
        # y-sharded views — no reshard, both from psi_yr_full.
        psi_yr = jax.lax.with_sharding_constraint(
            psi_yr_full, NamedSharding(mesh_xy, PSI_YR_SPEC))
        psi_yn = jax.lax.with_sharding_constraint(
            psi_yr_full.transpose(0, 2, 3, 1),
            NamedSharding(mesh_xy, PSI_YN_SPEC))

        if psi_xn_full is not None:
            # Zeta kernel layout: (nk, μ_X, nb, ns) — x on axis 1.
            # Transpose into the two canonical layouts; each transpose
            # preserves the x-sharding on the μ axis (see PSI_*_SPEC).
            psi_xn = jax.lax.with_sharding_constraint(
                psi_xn_full.transpose(0, 3, 1, 2),   # → (nk, ns, μ_X, nb)
                NamedSharding(mesh_xy, PSI_XN_SPEC))
            psi_xr = jax.lax.with_sharding_constraint(
                psi_xn_full.transpose(0, 2, 3, 1),   # → (nk, nb, ns, μ_X)
                NamedSharding(mesh_xy, PSI_XR_SPEC))
        else:
            # Fallback: derive x-sharded views from psi_yr_full (two
            # y→x all-to-alls on the μ axis).
            psi_xn = jax.lax.with_sharding_constraint(
                psi_yr_full.transpose(0, 2, 3, 1),
                NamedSharding(mesh_xy, PSI_XN_SPEC))
            psi_xr = jax.lax.with_sharding_constraint(
                psi_yr_full, NamedSharding(mesh_xy, PSI_XR_SPEC))
    return psi_xn, psi_xr, psi_yr, psi_yn


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
    psi_l_yr, psi_r_yr, *, enk_full, slices, mesh_xy, efermi=None,
    psi_l_xn=None, psi_r_xn=None,
) -> Wavefunctions:
    """Build all four sharded copies from left/right zeta-fit outputs.

    ``psi_{l,r}_yr`` are the required y-sharded halves.  Pass the
    x-sharded halves ``psi_{l,r}_xn`` too when available (the ISDF
    zeta-fit kernel returns both by construction) — that lets
    :func:`_build_four_copies` skip the y→x reshard on the μ axis.
    """
    psi_yr_full = _assemble_full_yr(psi_l_yr, psi_r_yr, slices)
    psi_xn_full = (
        _assemble_full_xn(psi_l_xn, psi_r_xn, slices)
        if (psi_l_xn is not None and psi_r_xn is not None)
        else None)
    return _finalise_bundle(
        psi_yr_full, psi_xn_full=psi_xn_full,
        enk_full=enk_full, slices=slices, mesh_xy=mesh_xy, efermi=efermi)


def build_wavefunctions_from_full(
    psi_yr_full, *, enk_full, slices, mesh_xy, efermi=None, psi_xn_full=None,
) -> Wavefunctions:
    """Build all four copies from already-assembled full-band arrays.

    Restart path: the HDF5-reloaded psi arrays are pre-assembled.  Same
    reshard-skip trick applies when psi_xn_full is provided.
    """
    return _finalise_bundle(
        psi_yr_full, psi_xn_full=psi_xn_full,
        enk_full=enk_full, slices=slices, mesh_xy=mesh_xy, efermi=efermi)


def _finalise_bundle(psi_yr_full, *, psi_xn_full, enk_full, slices, mesh_xy,
                      efermi):
    """Shared tail: four sharded psi views + occupations → Wavefunctions."""
    occ_full = _build_occ(enk_full, slices, efermi)
    psi_xn, psi_xr, psi_yr, psi_yn = _build_four_copies(
        psi_yr_full, mesh_xy, psi_xn_full=psi_xn_full)
    rep2 = NamedSharding(mesh_xy, P(None, None))
    with mesh_xy:
        enk_full = jax.lax.with_sharding_constraint(enk_full, rep2)
        occ_full = jax.lax.with_sharding_constraint(occ_full, rep2)
    return Wavefunctions(psi_xn=psi_xn, psi_xr=psi_xr, psi_yr=psi_yr,
                         psi_yn=psi_yn, enk=enk_full, occ=occ_full, slices=slices)
