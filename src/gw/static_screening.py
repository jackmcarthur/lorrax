"""Static RPA screening for consumers without a dynamical W model.

The response and head kernels remain owned by screening and qsgw_head.
The returned body is flat q, in orbit-packed centroid order, XY sharded;
its Gamma-cell head is a separate HeadSample in the usual Coulomb units.
Work is the existing O(N^3) ISDF response and Dyson solve. Large carriers
retain the existing two-dimensional processor distribution on CPU and GPU.
"""
from __future__ import annotations

import dataclasses
import os

import numpy as np

from .gw_config import HeadCorrection, LorraxConfig, QPSolver, infer_material_class


def build_static_screened_w(wfns, V_q, *, config, meta, mesh_xy, sym=None,
                            centroid_indices=None, head_resolver=None, occupation_state=None,
                            shared_pole=None,
                            print_fn=print):
    """Return W(0) and its matching q→0 head from a live or retained state.

    ``wfns`` contains the authenticated WFN centroid faces and energies in
    Ry. Without ``shared_pole`` this is a direct Dyson solve, with no pole
    fit. A supplied model is the evaluated QSGW/one-shot state and is read
    through its existing synthesis owner. ``V_q`` is the matching bare
    QirrOperator. The response body
    and optional local-field head fold use the same transition manifold.
    """
    if shared_pole is not None:
        # A retained model belongs to its evaluated map. Rebuilding from a
        # parent DFT WFN here would discard the QSGW state and change W.
        W = _shared_pole_static_body(
            shared_pole, V_q, meta=meta, mesh_xy=mesh_xy, print_fn=print_fn)
        return W, head_resolver.at(0.0j)
    if wfns is None:
        raise ValueError("Static Dyson screening requires wavefunction faces")
    from .minimax_screening import build_static_quadrature
    from .screening import ScreeningRequest, compute_screening
    from .qsgw_head import build_dft_head_response, finalize_iteration_head_samples

    quad, e_ref = (None, 0.0) if occupation_state is not None else build_static_quadrature(
        wfns, config.minimax_config, print_fn=print_fn)
    request = ScreeningRequest(0.0j, "static")
    response = None
    if config.head.correction is not HeadCorrection.OFF:
        response = build_dft_head_response(
            wfns, np.asarray([0.0j]), input_dir=config.input_dir,
            mesh=mesh_xy, wfn=head_resolver.wfn, meta=meta, config=config,
            wings=config.head.correction is HeadCorrection.FULL,
            occupation_state=occupation_state)
    roles = compute_screening(
        wfns, V_q, [request], quad=quad, e_ref=e_ref,
        sym=sym, centroid_indices=centroid_indices, config=config,
        meta=meta, mesh_xy=mesh_xy, print_fn=print_fn,
        iteration_head_response=response, occupation_state=occupation_state)
    W = roles["static"]
    if response is None:
        head = head_resolver.at(0.0j)
    else:
        head = finalize_iteration_head_samples(
            response, wfn=head_resolver.wfn, meta=meta, config=config,
            mesh=mesh_xy, requests=[request], W_by_role=roles).at(0.0j)
    return W, head


def build_static_w_from_restart(filename, input_file, mesh_xy, *, print_fn=print):
    """Build missing static screening from authenticated WFN/centroid state.

    The restart supplies the ISDF basis and bare Coulomb; no GW or Sigma is
    evaluated and the source restart is never mutated. Returned W is in
    canonical file centroid order, flat q with P(None, x, y), and includes
    no rank-one head (the BSE loader inserts the returned head once).
    """
    from wfn_loader import WfnLoader
    from file_io import load_centroid_basis
    from .gw_init import prepare_band_metadata, prepare_isdf_and_wavefunctions
    from .head_correction import HeadResolver
    from .restart_q_storage import take_pre_unfold

    if not input_file:
        raise ValueError("GATE bse_static_w_inputs: missing input_file; supply the WFN/centroid deck")
    config = LorraxConfig.from_input_file(input_file, print_fn=print_fn)
    if config.bispinor:
        raise ValueError(
            "GATE bse_static_w_bispinor_sectors: missing screened W0 on a "
            "four-current restart; the BSE direct kernel has no packed "
            "CC/CT/TC/TT response handoff. Supply an authenticated stored "
            "charge W0; a scalar rebuild would omit the coupled sectors.")
    if config.qp_solver is QPSolver.SELF_CONSISTENT:
        raise ValueError(
            "GATE bse_static_w_sc_state: missing final-map W0 on a QSGW "
            "restart; the parent WFN is the DFT state. Supply final-map W0 "
            "or a restart generated from WFN_qp.h5 with matching dipoles.")
    wfn = WfnLoader(config.paths.wfn_file, mesh=mesh_xy)
    material_class = infer_material_class(wfn.occs)
    sym = wfn.symmetry()
    basis = load_centroid_basis(config.paths.centroids_file, wfn.fft_grid, sym=sym)
    if not basis.orbit_closed:
        sym = sym.trivial_view()
    config = dataclasses.replace(config, restart=True, do_screened=True)
    meta, bands, _ = prepare_band_metadata(
        basis.centroid_indices, config, mesh_xy, basis.n_rmu, print_fn, sym, wfn)
    isdf = prepare_isdf_and_wavefunctions(
        cfg=config, wfn=wfn, sym=sym, meta=meta,
        centroid_indices=basis.centroid_indices, band_slices=bands,
        mesh_xy=mesh_xy, tmp_dir=os.path.dirname(filename),
        tensors_filename=filename, print0=print_fn)
    if isdf.wf_binding_charge is None:
        raise ValueError(
            "GATE bse_static_w_provenance: WFN/centroid receipt absent; "
            "regenerate the ISDF restart before rebuilding screening")
    from .efermi import solve_oneshot_occupations
    occupations = solve_oneshot_occupations(
        config, wfn, isdf.wf_bundle, material_class, mesh_xy=mesh_xy, print_fn=print_fn)
    resolver = HeadResolver(config, config.input_dir, wfn, sym, meta,
                            print_fn, mesh=mesh_xy)
    try:
        W, head = build_static_screened_w(
            isdf.wf_bundle, isdf.V_qmunu, config=config, meta=meta,
            mesh_xy=mesh_xy, sym=sym, centroid_indices=basis.centroid_indices,
            head_resolver=resolver, occupation_state=occupations, print_fn=print_fn)
        W = W.unfold(mesh_xy)
        W = meta.mu_basis.unpack_axis(W, -2)
        W = meta.mu_basis.unpack_axis(W, -1)
    finally:
        take_pre_unfold("W0_qmunu")
    print_fn("BSE: built static RPA W(0) and q->0 head from authenticated WFN + centroids")
    return W, head, float(meta.cell_volume)


def _shared_pole_static_body(shared_pole, V_q, *, meta, mesh_xy, print_fn):
    """Evaluate the retained map at zero, using the sole pole synthesis owner."""
    from symmetry_maps import QirrOperator
    from .mpa.sigma import shared_pole_static_wc
    from .restart_q_storage import deposit_pre_unfold

    V_op = QirrOperator.of(V_q)
    wc = shared_pole_static_wc(shared_pole, meta, mesh_xy=mesh_xy)
    if tuple(wc.shape) != (V_op.n_full, *V_op.values.shape[1:]):
        raise ValueError(
            "GATE shared_pole_static_w: model Wc(0) has shape "
            f"{tuple(wc.shape)}; V_q is {V_op.n_full} q x "
            f"{tuple(V_op.values.shape[1:])}; the two must share one "
            "packed centroid carrier")
    W0 = V_op.with_values(
        V_op.values + QirrOperator.whole_zone(wc).at_rows(V_op.full_rows))
    del wc
    if not V_op.is_whole_zone():
        deposit_pre_unfold(
            "W0_qmunu", W0.values, n_rmu_logical=int(meta.n_rmu),
            q_irr_frac=V_op.q_irr_frac, irr_idx_q=V_op.irr_idx,
            sym_idx_q=V_op.sym_idx, sym_perm=V_op.sym_perm,
            L_table=V_op.L_table, n_sym_spatial=V_op.n_sym_spatial,
            mu_basis=getattr(meta, "mu_basis", None))
    print_fn("  W0 (restart): V + Wc(omega = 0) of the shared-pole "
             f"model {str(shared_pole.get('digest', ''))[:12]}, on {V_op.n_wedge} of "
             f"{V_op.n_full} q")
    return W0
