from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace

import jax
import jax.numpy as jnp
from jax.sharding import Mesh


@dataclass(frozen=True)
class BandSlices:
    """Canonical band-edge bookkeeping and precomputed local slices."""

    b0: int
    b1: int
    b2: int
    b3: int
    b4: int
    full_slice: slice
    v_slice: slice
    c_slice: slice
    l_slice: slice
    r_slice: slice
    sigma_slice: slice
    occ_slice: slice
    coh_slice: slice

    @classmethod
    def from_band_edges(
        cls, b0: int, b1: int, b2: int, b3: int, b4: int
    ) -> "BandSlices":
        if not (b0 <= b1 <= b2 <= b3 <= b4):
            raise ValueError(
                f"Invalid band edges: expected b0<=b1<=b2<=b3<=b4, got {(b0, b1, b2, b3, b4)}"
            )
        full_stop = b4 - b0
        return cls(
            b0=b0,
            b1=b1,
            b2=b2,
            b3=b3,
            b4=b4,
            full_slice=slice(0, full_stop),
            v_slice=slice(0, b2 - b0),
            c_slice=slice(b2 - b0, b4 - b0),
            l_slice=slice(0, b3 - b0),
            r_slice=slice(b1 - b0, b4 - b0),
            sigma_slice=slice(b1 - b0, b3 - b0),
            occ_slice=slice(0, b2 - b0),
            coh_slice=slice(0, full_stop),
        )

    @property
    def nb_full(self) -> int:
        return self.b4 - self.b0

    @property
    def nb_l(self) -> int:
        return self.b3 - self.b0

    @property
    def nb_r(self) -> int:
        return self.b4 - self.b1


@dataclass
class WavefunctionBundle:
    """Two canonical wavefunction arrays plus replicated small metadata arrays.

    Shapes:
      psi_y: (nk, nb_full, nspinor, n_rmu), Y-sharded
      psi_x: (nk, nspinor, n_rmu, nb_full), X-sharded
      enk:   (nk, nb_full), replicated
      occ:   (nk, nb_full), replicated
    """

    psi_y: jax.Array
    psi_x: jax.Array
    enk: jax.Array
    occ: jax.Array
    slices: BandSlices

    def y(self, band_slice: slice) -> jax.Array:
        return self.psi_y[:, band_slice, :, :]

    def x(self, band_slice: slice) -> jax.Array:
        return self.psi_x[:, :, :, band_slice]


def _assemble_full_from_left_right(
    psi_l_y: jax.Array,
    psi_r_y: jax.Array | None,
    slices: BandSlices,
) -> jax.Array:
    """Build canonical b0:b4 Y-sharded array from left/right legacy arrays."""
    nb_full = slices.nb_full
    if psi_l_y.shape[1] >= nb_full:
        return psi_l_y[:, :nb_full, :, :]

    if psi_r_y is None:
        raise ValueError(
            "Cannot assemble full b0:b4 wavefunctions: psi_l does not contain full range and psi_r is missing."
        )

    nb_v = slices.b2 - slices.b0
    nb_c = slices.b4 - slices.b2
    r_c_start = slices.b2 - slices.b1

    if psi_l_y.shape[1] < nb_v:
        raise ValueError(
            f"psi_l has {psi_l_y.shape[1]} bands but needs at least {nb_v} for valence slice."
        )
    if psi_r_y.shape[1] < (r_c_start + nb_c):
        raise ValueError(
            f"psi_r has {psi_r_y.shape[1]} bands but needs at least {r_c_start + nb_c} to form conduction slice."
        )

    psi_v_y = psi_l_y[:, :nb_v, :, :]
    psi_c_y = psi_r_y[:, r_c_start : (r_c_start + nb_c), :, :]
    return jnp.concatenate([psi_v_y, psi_c_y], axis=1)


def build_wavefunction_bundle(
    psi_l_y: jax.Array,
    psi_r_y: jax.Array | None,
    *,
    enk_full: jax.Array,
    slices: BandSlices,
    mesh_xy: Mesh,
    sh: SimpleNamespace,
    efermi: float | None = None,
) -> WavefunctionBundle:
    """Construct canonical X/Y wavefunction storage plus replicated metadata."""
    psi_full_y = _assemble_full_from_left_right(psi_l_y, psi_r_y, slices)
    if psi_full_y.shape[1] != slices.nb_full:
        raise ValueError(
            f"Expected full band span {slices.nb_full}, got {psi_full_y.shape[1]}"
        )

    if enk_full.shape[1] != slices.nb_full:
        raise ValueError(
            f"enk_full must span b0:b4 ({slices.nb_full} bands), got {enk_full.shape[1]}"
        )

    if efermi is None:
        occ_full = jnp.zeros_like(enk_full, dtype=jnp.float64)
        occ_full = occ_full.at[:, slices.occ_slice].set(1.0)
    else:
        occ_full = jnp.asarray(enk_full <= efermi, dtype=jnp.float64)

    with mesh_xy:
        psi_full_y = jax.lax.with_sharding_constraint(psi_full_y, sh.y3_4)
        psi_full_x = jax.lax.with_sharding_constraint(
            psi_full_y.transpose(0, 2, 3, 1), sh.x2_4
        )
        enk_full = jax.lax.with_sharding_constraint(enk_full, sh.replicated_2)
        occ_full = jax.lax.with_sharding_constraint(occ_full, sh.replicated_2)

    return WavefunctionBundle(
        psi_y=psi_full_y,
        psi_x=psi_full_x,
        enk=enk_full,
        occ=occ_full,
        slices=slices,
    )


def build_wavefunction_bundle_from_full(
    psi_full_y: jax.Array,
    *,
    enk_full: jax.Array,
    slices: BandSlices,
    mesh_xy: Mesh,
    sh: SimpleNamespace,
    efermi: float | None = None,
    psi_full_x: jax.Array | None = None,
) -> WavefunctionBundle:
    """Construct canonical bundle from full-band arrays already spanning b0:b4."""
    if psi_full_y.shape[1] != slices.nb_full:
        raise ValueError(
            f"Expected psi_full_y to span b0:b4 ({slices.nb_full} bands), got {psi_full_y.shape[1]}"
        )
    if enk_full.shape[1] != slices.nb_full:
        raise ValueError(
            f"Expected enk_full to span b0:b4 ({slices.nb_full} bands), got {enk_full.shape[1]}"
        )

    if efermi is None:
        occ_full = jnp.zeros_like(enk_full, dtype=jnp.float64)
        occ_full = occ_full.at[:, slices.occ_slice].set(1.0)
    else:
        occ_full = jnp.asarray(enk_full <= efermi, dtype=jnp.float64)

    with mesh_xy:
        psi_full_y = jax.lax.with_sharding_constraint(psi_full_y, sh.y3_4)
        if psi_full_x is None:
            psi_full_x = jax.lax.with_sharding_constraint(
                psi_full_y.transpose(0, 2, 3, 1), sh.x2_4
            )
        else:
            psi_full_x = jax.lax.with_sharding_constraint(psi_full_x, sh.x2_4)
        enk_full = jax.lax.with_sharding_constraint(enk_full, sh.replicated_2)
        occ_full = jax.lax.with_sharding_constraint(occ_full, sh.replicated_2)

    return WavefunctionBundle(
        psi_y=psi_full_y,
        psi_x=psi_full_x,
        enk=enk_full,
        occ=occ_full,
        slices=slices,
    )


def build_sigma_views(
    wf_bundle: WavefunctionBundle,
    mesh_xy: Mesh,
    sh: SimpleNamespace,
) -> SimpleNamespace:
    """Build sigma-kernel wavefunction views from canonical bundle.

    Returns a namespace with psi_l, psi_lT, psi_coh, psi_cohT, psi_proj, psi_projT
    ready for the sigma pipeline.
    """
    s = wf_bundle.slices
    with mesh_xy:
        psi_l = jax.lax.with_sharding_constraint(wf_bundle.y(s.l_slice), sh.Y_shard)
        psi_lT = jax.lax.with_sharding_constraint(wf_bundle.x(s.l_slice), sh.XT_shard)
        psi_coh = jax.lax.with_sharding_constraint(wf_bundle.y(s.coh_slice), sh.Y_shard)
        psi_cohT = jax.lax.with_sharding_constraint(wf_bundle.x(s.coh_slice), sh.XT_shard)
        psi_proj = jax.lax.with_sharding_constraint(
            wf_bundle.x(s.l_slice).transpose(0, 3, 1, 2), sh.X_shard
        )
        psi_projT = jax.lax.with_sharding_constraint(
            wf_bundle.y(s.l_slice).transpose(0, 2, 3, 1), sh.YT_shard
        )
    return SimpleNamespace(
        psi_l=psi_l,
        psi_lT=psi_lT,
        psi_coh=psi_coh,
        psi_cohT=psi_cohT,
        psi_proj=psi_proj,
        psi_projT=psi_projT,
    )


