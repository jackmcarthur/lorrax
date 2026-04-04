from __future__ import annotations

import jax.numpy as jnp

from .types import PoleBlock


def build_wavefunction_poleblock(wf_bundle, band_slice: slice, *, label: str) -> PoleBlock:
    """Create a PoleBlock view from a WavefunctionBundle slice."""
    psi_x = wf_bundle.x(band_slice)
    psi_y = wf_bundle.y(band_slice)
    energies = wf_bundle.enk[:, band_slice]
    mask = jnp.ones_like(energies, dtype=bool)
    return PoleBlock(
        psi_X=psi_x,
        psi_Y=psi_y,
        energies=energies,
        mask=mask,
        label=label,
    )


def valence_poleblock_from_bundle(wf_bundle) -> PoleBlock:
    """Build valence PoleBlock from canonical bundle slices."""
    return build_wavefunction_poleblock(
        wf_bundle,
        wf_bundle.slices.v_slice,
        label="valence_wfns",
    )


def conduction_poleblock_from_bundle(wf_bundle) -> PoleBlock:
    """Build conduction PoleBlock from canonical bundle slices."""
    return build_wavefunction_poleblock(
        wf_bundle,
        wf_bundle.slices.c_slice,
        label="conduction_wfns",
    )
