"""Bare transverse exchange Σ^B = −Σ_ij γ̃^i G γ̃^j V^{ij}, with i,j=1,2,3."""
from __future__ import annotations

from pathlib import Path

import jax
import jax.numpy as jnp
from jax.sharding import Mesh

from .v_q_bispinor import BispinorVqReader
from .wavefunction_bundle import padded_centroid_extent


_TRANSVERSE_INDICES = (1, 2, 3)


def compute_sigma_x_bispinor(
    *, wfns_transverse, Gij: jax.Array, bispinor_v_q_path: Path | str,
    meta, mesh_xy: Mesh, print_fn=print, verbose: bool = True,
) -> jax.Array:
    """Sum the nine TT exchange tiles through the photon block contraction."""
    from .cohsex_sigma import _replicate_band_sigma
    from .photon_sigma import contract_lorentz_blocks, _TERM_X

    extent = padded_centroid_extent(wfns_transverse)
    with BispinorVqReader(bispinor_v_q_path, mesh_xy) as reader:
        def get_block(A, B):
            V = reader.get_tile(A, B)
            if V.shape[-2:] != (extent, extent):
                V = jnp.pad(V, ((0, 0), (0, extent - V.shape[-2]),
                                (0, extent - V.shape[-1])))
            return V, V, V, V

        sig, _, _, _ = contract_lorentz_blocks(
            [(A, B) for A in _TRANSVERSE_INDICES for B in _TRANSVERSE_INDICES],
            carrier_C=wfns_transverse, carrier_T=wfns_transverse,
            plan_C=None, plan_T=None, term=(_TERM_X,), mesh_xy=mesh_xy,
            meta=meta, Gij=Gij, get_block=get_block,
            print_fn=print_fn, verbose=verbose)
    result = _replicate_band_sigma(sig[_TERM_X], mesh_xy)
    if wfns_transverse.layout == "face":
        nb = wfns_transverse.slices.nb_sigma
        result = result[:, :nb, :nb]
    result.block_until_ready()
    return result
