"""Bare transverse exchange through the shared parent Lorentz contraction."""
from __future__ import annotations

import hashlib
import os

import numpy as np

#: The packed bare transverse V of the run, held on the host between SC maps
#: (survey B5): the V file is immutable for the run, and re-reading it cost
#: 16 SlabIO reads of 3.3 GB per VI3 P100 map. One entry, keyed on the file
#: generation, the transverse basis and the packing, so any change re-reads.
_HOST_PACKED = {}


def _packed_key(path, plan, basis, layout, mesh_xy):
    stat = os.stat(path)
    digest = hashlib.sha256()
    for array in (plan.sym_perm, plan.L_table, basis.canonical_indices):
        digest.update(np.ascontiguousarray(array).tobytes())
    return (os.path.realpath(path), stat.st_size, stat.st_mtime_ns,
            digest.hexdigest(), int(layout.packed_extent),
            tuple(int(device.id) for device in mesh_xy.devices.flat))


def _bare_transverse_packed(path, *, plan, mu_bases, layout, mesh_xy):
    """Read, authenticate and pack the nine TT tiles once; restore thereafter.

    Blocks with a charge index are exact zeros: the packer leaves them at its
    zero initial value, so they are never read (the previous route read the
    (1,1) tile seven more times per map only to take its shape).
    """
    from common.collectives import restore_from_host, spill_to_host
    from file_io.restart_bundle import BispinorVqReader
    from .photon_layout import pack_photon_operator
    key = _packed_key(path, plan, mu_bases[1], layout, mesh_xy)
    if key not in _HOST_PACKED:
        with BispinorVqReader(path, mesh_xy, mu_bases=mu_bases,
                              family_plans=(None, plan)) as reader:
            packed = pack_photon_operator(
                lambda a, b: reader.get_tile(a, b) if a and b else None,
                reader.n_q_total, layout, mesh_xy)
        _HOST_PACKED.clear()
        _HOST_PACKED[key] = spill_to_host(packed)
    return restore_from_host(_HOST_PACKED[key])


def compute_sigma_x_bispinor(
    *, wfns_transverse, Gij, bispinor_v_q_path, meta, mesh_xy, mu_bases,
    print_fn=print, verbose=True,
):
    """Sum the nine TT bare exchange blocks on parents, then unfold the band operator."""
    from symmetry_maps import unfold_file_wedge_band_operator
    from .cohsex_sigma import _replicate_band_sigma
    from .photon_layout import PhotonBasisLayout
    from .photon_sigma import contract_lorentz_blocks, _TERM_X
    from .w_isdf import StaticPhotonResponse
    from .qgrid_symmetry import qgrid_trs_policy_for
    plan = wfns_transverse.green_parent.plan
    sym = plan.sym
    policy = qgrid_trs_policy_for(sym=sym, irr_idx_q=sym.irr_idx_q,
        sym_idx_q=sym.sym_idx_q, kgrid=tuple(meta.kgrid),
        n_sym_spatial=plan.n_sym_spatial, context="bare transverse Sigma")
    extent = plan.n_centroid_packed
    layout = PhotonBasisLayout.from_centroid_extents(extent, extent, mesh_xy)
    packed = _bare_transverse_packed(bispinor_v_q_path, plan=plan,
                                     mu_bases=mu_bases, layout=layout,
                                     mesh_xy=mesh_xy)
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
