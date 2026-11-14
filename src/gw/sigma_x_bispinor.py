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
    # Diagnostic switches (common.bispinor_diag); both inert when unset.
    from common.bispinor_diag import tt_exchange_enabled, tt_exchange_skip_bands
    skip_bands = tt_exchange_skip_bands()
    if skip_bands:
        Gij = _drop_lowest_occupied(Gij, skip_bands, wfns_transverse.slices)
    sigma = None
    for key, value, _ in contract_lorentz_blocks(
            [(a,b) for a in (1,2,3) for b in (1,2,3)],
            families=(wfns_transverse,wfns_transverse), term=_TERM_X,
            response=response, Gij=Gij, meta=meta, mesh_xy=mesh_xy):
        sigma = value if sigma is None else sigma + value
    sigma = unfold_file_wedge_band_operator(sym, sigma, trs_rule="transpose")
    sigma = _replicate_band_sigma(sigma, mesh_xy)[:, :wfns_transverse.slices.nb_sigma,
                                                 :wfns_transverse.slices.nb_sigma]
    if not tt_exchange_enabled():
        # LORRAX_DIAG_TT_EXCHANGE=off: the contraction ran, so shape, dtype
        # and sharding are the production ones; only the value is removed.
        sigma = sigma * 0.0
    return sigma


def _drop_lowest_occupied(Gij, n_skip, slices):
    """``Gij`` with the ``n_skip`` lowest bands of the Sigma window unoccupied.

    LORRAX_DIAG_TT_EXCHANGE_SKIP_BANDS only.  ``Gij`` is the diagonal
    occupation projector ``(nk, nb_sigma, nb_sigma)`` on the Sigma window's
    local band index (global band ``slices.b0 + i``); the contraction reads its
    diagonal as the occupied weights (``cohsex_sigma._occ_diag_full``), so
    zeroing rows ``[0, n_skip)`` removes those intermediate states from the
    exchange sum and nothing else.  The mask is a compile-time constant and
    the result keeps the input's sharding.
    """
    import jax
    nb = int(Gij.shape[-1])
    n_skip = int(n_skip)
    if not 0 < n_skip < nb:
        raise ValueError(
            f"LORRAX_DIAG_TT_EXCHANGE_SKIP_BANDS={n_skip} must lie in "
            f"(0, {nb}), the Sigma window's band count")
    keep = np.ones((1, nb, 1), dtype=np.float64)
    keep[:, :n_skip, :] = 0.0
    print(f"*** LORRAX DIAG: Sigma^B occupied sum drops Sigma-window bands "
          f"[0, {n_skip}) = global bands [{int(slices.b0)}, "
          f"{int(slices.b0) + n_skip})", flush=True)
    sharding = getattr(Gij, "sharding", None)
    mask = jax.jit(lambda g: g * keep) if sharding is None else jax.jit(
        lambda g: g * keep, out_shardings=sharding)
    return mask(Gij)
