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
    r"""Bare transverse exchange $\Sigma^B = -\sum_{ij}\alpha^i G \alpha^j t_{ij}$, $t = -vP^T$.

    The nine TT blocks are summed on parents and the band operator unfolded
    (docs/theory/bispinor-gw.md section 4).  Two deck keys act here and
    nowhere else, through ``meta`` (set by
    :func:`gw.gw_init.prepare_band_metadata`):

    * ``meta.transverse_exchange`` False (``bispinor_transverse_exchange =
      false``): no contraction runs and the result is an exact zero of the
      production shape, dtype and sharding, so ``sigTT`` is 0 and Sigma is
      the charge part.
    * ``meta.transverse_exchange_skip_bands`` N > 0: global bands [0, N)
      (the Sigma window starts at band 0) leave the occupied sum through a
      masked copy of ``Gij``; the charge exchange, the density and the
      current keep them (:func:`_drop_lowest_occupied`).

    Returns
    -------
    jax.Array
        ``(nk_tot, nb_sigma, nb_sigma)`` complex, Ry, replicated
        ``P(None, None, None)`` on ``mesh_xy``.
    """
    from symmetry_maps import unfold_file_wedge_band_operator
    from .cohsex_sigma import _replicate_band_sigma
    from .photon_layout import PhotonBasisLayout
    from .photon_sigma import contract_lorentz_blocks, _TERM_X
    from .w_isdf import StaticPhotonResponse
    from .qgrid_symmetry import qgrid_trs_policy_for
    nb_sigma = int(wfns_transverse.slices.nb_sigma)
    if not bool(getattr(meta, "transverse_exchange", True)):
        import jax
        import jax.numpy as jnp
        from jax.sharding import NamedSharding, PartitionSpec as P
        if verbose:
            print_fn("  Sigma^B: bispinor_transverse_exchange = false; the bare "
                     "transverse exchange is not computed (exact zero)")
        return jax.jit(
            lambda: jnp.zeros((int(meta.nk_tot), nb_sigma, nb_sigma),
                              dtype=jnp.complex128),
            out_shardings=NamedSharding(mesh_xy, P(None, None, None)))()
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
    Gij = transverse_exchange_occupations(
        Gij, meta, wfns_transverse.slices,
        print_fn=print_fn if verbose else None)
    sigma = None
    for key, value, _ in contract_lorentz_blocks(
            [(a,b) for a in (1,2,3) for b in (1,2,3)],
            families=(wfns_transverse,wfns_transverse), term=_TERM_X,
            response=response, Gij=Gij, meta=meta, mesh_xy=mesh_xy):
        sigma = value if sigma is None else sigma + value
    sigma = unfold_file_wedge_band_operator(sym, sigma, trs_rule="transpose")
    return _replicate_band_sigma(sigma, mesh_xy)[:, :nb_sigma, :nb_sigma]


def transverse_exchange_occupations(Gij, meta, slices, *, print_fn=None):
    """The occupied weights of Sigma^B's intermediate-state sum: ONE owner.

    Every semicore treatment of the transverse exchange that acts on its
    occupied sum lives here, so a later branch (a shell projector, another
    vertex) changes this function and not the key handling
    (``gw.gw_config.bispinor_transverse_terms_stamp`` records which
    treatment a run used).  Today: with ``meta.transverse_exchange_skip_bands``
    N > 0 the rows ``[0, N)`` of ``Gij`` are zeroed; otherwise ``Gij`` is
    returned unchanged (the same object).

    Parameters
    ----------
    Gij : jax.Array
        ``(nk, nb_sigma, nb_sigma)`` float64 diagonal occupation projector on
        the Sigma window's local band index (global band ``slices.b0 + i``),
        any sharding.  The contraction reads its diagonal as the occupied
        weights (``cohsex_sigma._occ_diag_full``), so a zero row removes that
        intermediate state from the exchange sum and nothing else.  On an SC
        map the rows are the map's QP states.
    meta : common.Meta
        Carries ``transverse_exchange_skip_bands`` (set by ``gw_init``).
    slices : gw.wavefunction_bundle.BandSlices
        For the printed global band range.
    print_fn : callable or None
        One log line when a treatment acts.

    Returns
    -------
    jax.Array
        Same shape, dtype and sharding as ``Gij``; the input is not modified.
    """
    n_skip = int(getattr(meta, "transverse_exchange_skip_bands", 0))
    if n_skip == 0:
        return Gij
    import jax
    nb = int(Gij.shape[-1])
    if not 0 < n_skip < nb:
        raise ValueError(
            "GATE bispinor_transverse_exchange_skip_bands: "
            f"got {n_skip}; want a count in (0, {nb}), the Sigma window's "
            "band count")
    keep = np.ones((1, nb, 1), dtype=np.float64)
    keep[:, :n_skip, :] = 0.0
    if print_fn is not None:
        print_fn(f"  Sigma^B: occupied sum without Sigma-window bands "
                 f"[0, {n_skip}) = global bands [{int(slices.b0)}, "
                 f"{int(slices.b0) + n_skip}) "
                 "(bispinor_transverse_exchange_skip_bands)")
    sharding = getattr(Gij, "sharding", None)
    mask = jax.jit(lambda g: g * keep) if sharding is None else jax.jit(
        lambda g: g * keep, out_shardings=sharding)
    return mask(Gij)
