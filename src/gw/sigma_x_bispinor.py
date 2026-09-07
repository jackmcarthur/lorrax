"""Bare transverse exchange through the shared parent Lorentz contraction."""
from __future__ import annotations

import jax.numpy as jnp


def compute_sigma_x_bispinor(
    *, wfns_transverse, Gij, bispinor_v_q_path, meta, mesh_xy, mu_bases,
    print_fn=print, verbose=True,
):
    """Sum the nine TT bare exchange blocks on parents, then unfold the band operator."""
    from symmetry_maps import unfold_file_wedge_band_operator
    from .cohsex_sigma import _replicate_band_sigma
    from .photon_layout import PhotonBasisLayout, pack_photon_operator
    from .photon_sigma import contract_lorentz_blocks, _TERM_X
    from .v_q_bispinor import BispinorVqReader
    from .w_isdf import StaticPhotonResponse
    from .qgrid_symmetry import qgrid_trs_policy_for
    plan = wfns_transverse.green_parent.plan
    sym = plan.sym
    policy = qgrid_trs_policy_for(sym=sym, irr_idx_q=sym.irr_idx_q,
        sym_idx_q=sym.sym_idx_q, kgrid=tuple(meta.kgrid),
        n_sym_spatial=plan.n_sym_spatial, context="bare transverse Sigma")
    extent = plan.n_centroid_packed
    layout = PhotonBasisLayout.from_centroid_extents(extent, extent, mesh_xy)
    with BispinorVqReader(bispinor_v_q_path, mesh_xy, mu_bases=mu_bases,
                         family_plans=(None, plan)) as reader:
        def tile(a,b):
            if a and b:
                return reader.get_tile(a,b)
            return jnp.zeros_like(reader.get_tile(1,1))
        packed = pack_photon_operator(tile, reader.n_q_total, layout, mesh_xy)
    response = StaticPhotonResponse(layout, packed, packed, "none", "bare_transverse",
        qgrid_policy=policy, family_plans=(plan,plan))
    sigma = None
    for key, value, _ in contract_lorentz_blocks(
            [(a,b) for a in (1,2,3) for b in (1,2,3)],
            families=(wfns_transverse,wfns_transverse), term=_TERM_X,
            response=response, Gij=Gij, meta=meta, mesh_xy=mesh_xy):
        sigma = value if sigma is None else sigma + value
    sigma = unfold_file_wedge_band_operator(sym, sigma, trs_rule="transpose")
    return _replicate_band_sigma(sigma, mesh_xy)[:, :wfns_transverse.slices.nb_sigma,
                                                :wfns_transverse.slices.nb_sigma]
