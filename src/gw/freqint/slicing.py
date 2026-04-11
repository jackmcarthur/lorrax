from __future__ import annotations

import jax.numpy as jnp

from .types import PoleBlock


def build_wavefunction_poleblock(wf_bundle, band_slice: slice, *, label: str) -> PoleBlock:
    """Create a PoleBlock view from a Wavefunctions bundle slice."""
    psi_xn = wf_bundle.xn(band_slice)
    psi_yr = wf_bundle.yr(band_slice)
    energies = wf_bundle.enk[:, band_slice]
    mask = jnp.ones_like(energies, dtype=bool)
    return PoleBlock(
        psi_X=psi_xn,
        psi_Y=psi_yr,
        energies=energies,
        mask=mask,
        label=label,
    )


def valence_poleblock_from_bundle(wf_bundle) -> PoleBlock:
    """Build valence PoleBlock from canonical bundle slices."""
    return build_wavefunction_poleblock(
        wf_bundle,
        wf_bundle.slices.val,
        label="valence_wfns",
    )


def conduction_poleblock_from_bundle(wf_bundle) -> PoleBlock:
    """Build conduction PoleBlock from canonical bundle slices."""
    return build_wavefunction_poleblock(
        wf_bundle,
        wf_bundle.slices.cond,
        label="conduction_wfns",
    )
