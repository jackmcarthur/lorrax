"""Blockwise static self-energy for a screened four-current propagator.

The response owner stores ``D^{AB}`` as one packed, two-dimensionally sharded
operator.  This module only schedules its sixteen rectangular block views and
contracts them.  The physics kernels remain the existing owners:

* :func:`gw.greens_function_kernel.build_G` builds the rectangular Green
  function;
* :func:`gw.cohsex_sigma._make_static_convolution` performs the flat-k FFT
  convolution; and
* :func:`common.contract_bands.contract_bands_block_reshard` projects the
  exchange/correlation operator back to band space.

The accumulator stays 2-D sharded until the ordinary static-Sigma result
boundary (the face carrier first gathers its canonical full-band result, then
windows it exactly as scalar COHSEX does).  A photon body or Green tensor is
never gathered or held beside another block.
"""

from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

from common.collectives import device_put_process_local


_CHANNELS = range(4)
_TERM_X = 0
_TERM_SX = 1
_TERM_COH = 2
_HEAD_CC = 0
_HEAD_CTTC = 1
_HEAD_TT = 2
_photon_sigma_kernel_cache: dict[tuple[object, ...], object] = {}


@dataclass(frozen=True)
class StaticPhotonHeadSigmaDiagnostics:
    """Exact diagonal contraction of the final q=0 Lorentz-block updates.

    ``components_tskn_ry`` has term order ``(X,SX,COH)`` and sector order
    ``(CC,CT+TC,TT)``.  The sectors classify FINAL completed V/W blocks;
    they are not nonlinear counterfactual Dyson solves.
    """

    components_tskn_ry: jax.Array
    max_closure_residual_ry: float
    hamiltonian_config_operator_fingerprint: str | None
    output_basis: str

    def __post_init__(self) -> None:
        shape = tuple(self.components_tskn_ry.shape)
        if len(shape) != 4 or shape[:2] != (3, 3):
            raise ValueError(
                "static photon head diagnostics must be (3 terms,3 sectors,"
                f"nk,nb); got {shape}")
        if self.hamiltonian_config_operator_fingerprint is not None:
            from .head_correction import require_canonical_operator_fingerprint
            fingerprint = require_canonical_operator_fingerprint(
                self.hamiltonian_config_operator_fingerprint,
                gate="static_photon_head_sigma_fingerprint",
            )
            object.__setattr__(
                self, "hamiltonian_config_operator_fingerprint", fingerprint)
        if self.output_basis != "dft":
            raise ValueError(
                "static photon head Sigma diagnostics must be stamped in "
                f"the DFT output basis; got {self.output_basis!r}")


def _head_sector(A: int, B: int) -> int:
    if int(A) == 0 and int(B) == 0:
        return _HEAD_CC
    if (int(A) == 0) != (int(B) == 0):
        return _HEAD_CTTC
    return _HEAD_TT


def _sigma_window_matrix(matrix, wfns_charge):
    if wfns_charge.layout != "face":
        return matrix
    nb_sigma = int(wfns_charge.slices.nb_sigma)
    return matrix[..., :nb_sigma, :nb_sigma]


def _diagnostic_diagonal(matrix, basis_rotation, mesh_xy):
    """Exact output-basis diagonal for ``(nk,...,nb,nb)`` batches."""
    if basis_rotation is None:
        diag = jnp.diagonal(matrix, axis1=-2, axis2=-1)
    else:
        from .qsgw_density import diagonal_rotated_band_matrix
        diag = diagonal_rotated_band_matrix(
            matrix, basis_rotation, mesh=mesh_xy, to_qp=False)
    # Diagnostics are O(nk*nb), so their public carrier is deliberately
    # replicated.  No dense rotated nb^2 matrix crosses this boundary.
    return jax.lax.with_sharding_constraint(
        diag, NamedSharding(mesh_xy, P(*([None] * diag.ndim))))


def _padded_centroid_extent(wfns) -> int:
    if wfns.layout == "legacy":
        return int(wfns.psi_yr.shape[-1])
    if wfns.layout == "face":
        return int(wfns.psi_mun.shape[2])
    raise ValueError(f"photon Sigma: unknown wavefunction layout {wfns.layout!r}")


def _bundle_for_channel(wfns_charge, wfns_transverse, channel: int):
    return wfns_charge if int(channel) == 0 else wfns_transverse


def _require_packed_operator(name, packed, mesh_xy):
    expected = NamedSharding(mesh_xy, P(None, "x", "y"))
    have = packed.sharding
    if (getattr(have, "mesh", None) != expected.mesh
            or getattr(have, "spec", None) != expected.spec):
        raise ValueError(
            f"photon operator {name} must remain P(None,'x','y'); got "
            f"{packed.sharding}.  A photon body may not be gathered or "
            "placed on fewer than all ranks.")


def _make_photon_static_block_kernel(
    mesh_xy: Mesh, kgrid, nk_tot: int, wfns_left, wfns_right, *,
    with_q0_diagnostic: bool,
):
    """One block contraction, specialized only by endpoint carrier shapes.

    The Lorentz matrices are folded into the two bundles outside this kernel
    by :func:`gw.wavefunction_bundle.with_lorentz_vertices`; their channel
    numbers therefore never become static kernel arguments.  ``term`` is a
    dynamic selector so X, SX, and COH share one executable while executing
    one Green/operator tile at a time.
    """
    from ffi import ffi_dial_key

    from .wavefunction_bundle import face_kernel_kwargs

    endpoint = face_kernel_kwargs(wfns_left, wfns_right)
    key = (id(mesh_xy), tuple(int(v) for v in kgrid), int(nk_tot),
           ffi_dial_key(), wfns_left.layout,
           endpoint.get("face_shape"), endpoint.get("right_face_shape"),
           bool(with_q0_diagnostic))
    if key in _photon_sigma_kernel_cache:
        return _photon_sigma_kernel_cache[key]

    from common.contract_bands import contract_bands_block_reshard
    from .cohsex_sigma import _make_static_convolution, _occ_diag_full
    from .greens_function_kernel import build_G

    convolve = _make_static_convolution(mesh_xy, kgrid, nk_tot)
    convolve_q0 = (
        _make_static_convolution(
            mesh_xy, kgrid, nk_tot, q0_only=True)
        if with_q0_diagnostic else None)
    project = contract_bands_block_reshard(mesh_xy, **endpoint)

    if wfns_left.layout == "face":
        from distrib_la import gemm_plan
        _, nb_full, mu_left, ns = endpoint["face_shape"]
        right_shape = endpoint.get("right_face_shape", endpoint["face_shape"])
        _, nb_right, mu_right, ns_right = right_shape
        if (nb_right, ns_right) != (nb_full, ns):
            raise ValueError(
                "photon Sigma face endpoints must share band/spin extents; "
                f"got {endpoint['face_shape']} and {right_shape}")
        g_plan = gemm_plan(
            mesh_xy, m=mu_left * ns, k=nb_full, n=mu_right * ns,
            nq=int(nk_tot), dtype=jnp.complex128)
    else:
        g_plan = None

    @jax.jit
    def contract_block(wfns_left, wfns_right, wfns_left_g, wfns_right_g,
                       Gij, W_AB, V_AB, W_head_AB, V_head_AB, term):
        s_left = wfns_left.slices
        s_right = wfns_right.slices

        if wfns_left.layout == "legacy":
            def occupied(interactions):
                interaction, head_interaction = interactions
                G = build_G(
                    wfns_left_g.xn(s_left.sigma),
                    wfns_right_g.yr(s_right.sigma), Gij=Gij)
                O = convolve(G, interaction, 1.0)
                result = project(
                    wfns_left.xr(s_left.sigma), O,
                    wfns_right.yn(s_right.sigma))
                if with_q0_diagnostic:
                    O_head = convolve_q0(G, head_interaction, 1.0)
                    head_result = project(
                        wfns_left.xr(s_left.sigma), O_head,
                        wfns_right.yn(s_right.sigma))
                else:
                    head_result = result
                return result, head_result

            def coh(_):
                G = build_G(
                    wfns_left_g.xn(s_left.sigma_sum),
                    wfns_right_g.yr(s_right.sigma_sum))
                O = convolve(G, W_AB - V_AB, -0.5)
                result = project(
                    wfns_left.xr(s_left.sigma), O,
                    wfns_right.yn(s_right.sigma))
                if with_q0_diagnostic:
                    O_head = convolve_q0(
                        G, W_head_AB - V_head_AB, -0.5)
                    head_result = project(
                        wfns_left.xr(s_left.sigma), O_head,
                        wfns_right.yn(s_right.sigma))
                else:
                    head_result = result
                return result, head_result
        else:
            occ = _occ_diag_full(
                Gij, s_left.nb_sigma, s_left.nb_full)
            ri_mask = wfns_left.band_mask(
                s_left.sigma_sum).astype(jnp.complex128)

            def occupied(interactions):
                interaction, head_interaction = interactions
                G = build_G(
                    wfns_left_g.psi_mun, wfns_right_g.psi_nmu,
                    phases=occ, layout="face", gemm=g_plan)
                O = convolve(G, interaction, 1.0)
                result = project(
                    wfns_left.psi_nmu, O, wfns_right.psi_mun)
                if with_q0_diagnostic:
                    O_head = convolve_q0(G, head_interaction, 1.0)
                    head_result = project(
                        wfns_left.psi_nmu, O_head, wfns_right.psi_mun)
                else:
                    head_result = result
                return result, head_result

            def coh(_):
                G = build_G(
                    wfns_left_g.psi_mun, wfns_right_g.psi_nmu,
                    phases=ri_mask, layout="face", gemm=g_plan)
                O = convolve(G, W_AB - V_AB, -0.5)
                result = project(
                    wfns_left.psi_nmu, O, wfns_right.psi_mun)
                if with_q0_diagnostic:
                    O_head = convolve_q0(
                        G, W_head_AB - V_head_AB, -0.5)
                    head_result = project(
                        wfns_left.psi_nmu, O_head, wfns_right.psi_mun)
                else:
                    head_result = result
                return result, head_result

        def x_or_sx(_):
            interactions = jax.lax.cond(
                term == _TERM_X,
                lambda __: (V_AB, V_head_AB),
                lambda __: (W_AB, W_head_AB),
                operand=None,
            )
            return occupied(interactions)

        result, head_result = jax.lax.cond(
            term == _TERM_COH, coh, x_or_sx, operand=None)
        if with_q0_diagnostic:
            return result, head_result
        return result

    _photon_sigma_kernel_cache[key] = contract_block
    return contract_block


def compute_static_photon_sigma(
    *,
    wfns_charge,
    wfns_transverse,
    Gij: jax.Array,
    V_packed: jax.Array,
    W_packed: jax.Array,
    photon_layout,
    meta,
    mesh_xy: Mesh,
    head_completion=None,
    diagnostic_basis_rotation=None,
    diagnostic_input_basis=None,
    print_fn=print,
    verbose: bool = True,
) -> tuple[
    jax.Array, jax.Array, jax.Array, StaticPhotonHeadSigmaDiagnostics | None,
]:
    """Stream all sixteen ``D^{AB}`` blocks into full static COHSEX.

    ``V_packed`` and ``W_packed`` must stay at ``P(None, 'x', 'y')``.
    :func:`gw.photon_layout.photon_block_view` returns a mesh-aligned padded
    view, whose two extents must equal those of the corresponding charge or
    transverse wavefunction bundle.  No logical block is copied or gathered.
    """
    if wfns_charge.layout != wfns_transverse.layout:
        raise ValueError(
            "photon Sigma requires charge and transverse wavefunction "
            f"bundles in one representation; got {wfns_charge.layout!r} and "
            f"{wfns_transverse.layout!r}.")
    if wfns_charge.slices != wfns_transverse.slices:
        raise ValueError(
            "photon Sigma requires the charge and transverse bundles to use "
            "the same band windows; their BandSlices records differ.")

    for name, packed in (("V_packed", V_packed), ("W_packed", W_packed)):
        _require_packed_operator(name, packed, mesh_xy)

    from .photon_layout import photon_block_view, photon_q0_low_rank_block

    q0_factors = (
        None if head_completion is None
        else getattr(head_completion, "q0_factors", None))
    if head_completion is not None and q0_factors is None:
        raise ValueError(
            "packed photon head completion lacks its bounded q0 factor "
            "carrier; refusing a decomposition inferred from the packed body")
    if q0_factors is not None:
        if diagnostic_input_basis not in ("dft", "qp"):
            raise ValueError(
                "photon head Sigma diagnostics require explicit input basis "
                f"'dft' or 'qp'; got {diagnostic_input_basis!r}")
        if ((diagnostic_input_basis == "qp")
                != (diagnostic_basis_rotation is not None)):
            raise ValueError(
                "photon head Sigma diagnostic basis/rotation mismatch: "
                f"input_basis={diagnostic_input_basis!r}, rotation="
                f"{'set' if diagnostic_basis_rotation is not None else 'None'}")

    sig_x = None
    sig_sx = None
    sig_coh = None
    head_diag = [[None for _ in range(3)] for _ in range(3)]
    head_total_diag = [None for _ in range(3)]

    from .wavefunction_bundle import with_lorentz_vertices

    for A in _CHANNELS:
        left = _bundle_for_channel(wfns_charge, wfns_transverse, A)
        left_g = with_lorentz_vertices(left, A, 0)
        n_left = _padded_centroid_extent(left)
        for B in _CHANNELS:
            right = _bundle_for_channel(wfns_charge, wfns_transverse, B)
            right_g = with_lorentz_vertices(right, 0, B)
            n_right = _padded_centroid_extent(right)
            contract_block = _make_photon_static_block_kernel(
                mesh_xy, meta.kgrid, int(meta.nk_tot), left, right,
                with_q0_diagnostic=q0_factors is not None)
            V_AB = photon_block_view(V_packed, photon_layout, A, B, mesh_xy)
            W_AB = photon_block_view(W_packed, photon_layout, A, B, mesh_xy)
            expected = (int(meta.nk_tot), n_left, n_right)
            if tuple(V_AB.shape) != expected or tuple(W_AB.shape) != expected:
                raise ValueError(
                    f"photon block ({A},{B}) shape mismatch: expected padded "
                    f"{expected} from its wavefunction endpoints, got "
                    f"V{tuple(V_AB.shape)} and W{tuple(W_AB.shape)}.")

            if q0_factors is not None:
                V_head_AB = photon_q0_low_rank_block(
                    (q0_factors.bare_pair,), photon_layout, A, B, mesh_xy)
                W_head_AB = photon_q0_low_rank_block(
                    q0_factors.screened_pairs, photon_layout, A, B, mesh_xy)
            else:
                # Closed-static false branch: these aliases are dead JIT
                # operands and allocate no second body/block.
                V_head_AB, W_head_AB = V_AB, W_AB

            x_result = contract_block(
                left, right, left_g, right_g, Gij, W_AB, V_AB,
                W_head_AB, V_head_AB,
                jnp.asarray(_TERM_X, dtype=jnp.int32))
            if q0_factors is None:
                x_AB, x_head_AB = x_result, None
            else:
                x_AB, x_head_AB = x_result
            sig_x = x_AB if sig_x is None else sig_x + x_AB
            sig_x.block_until_ready()
            sx_result = contract_block(
                left, right, left_g, right_g, Gij, W_AB, V_AB,
                W_head_AB, V_head_AB,
                jnp.asarray(_TERM_SX, dtype=jnp.int32))
            if q0_factors is None:
                sx_AB, sx_head_AB = sx_result, None
            else:
                sx_AB, sx_head_AB = sx_result
            sig_sx = sx_AB if sig_sx is None else sig_sx + sx_AB
            sig_sx.block_until_ready()
            coh_result = contract_block(
                left, right, left_g, right_g, Gij, W_AB, V_AB,
                W_head_AB, V_head_AB,
                jnp.asarray(_TERM_COH, dtype=jnp.int32))
            if q0_factors is None:
                coh_AB, coh_head_AB = coh_result, None
            else:
                coh_AB, coh_head_AB = coh_result
            sig_coh = coh_AB if sig_coh is None else sig_coh + coh_AB

            if q0_factors is not None:
                sector = _head_sector(A, B)
                # Batch the orthogonal X/SX/COH terms through one canonical
                # diagonal rotation.  This is one half-rotation per Lorentz
                # block, not three full dense U A U^dagger materialisations.
                contributions = jnp.stack(
                    (x_head_AB, sx_head_AB, coh_head_AB), axis=1)
                contributions = _sigma_window_matrix(
                    contributions, wfns_charge)
                diagonal_tkn = jnp.moveaxis(_diagnostic_diagonal(
                    contributions, diagnostic_basis_rotation, mesh_xy), 1, 0)
                diagonal_tkn.block_until_ready()
                for term in range(3):
                    diagonal = diagonal_tkn[term]
                    previous = head_diag[term][sector]
                    head_diag[term][sector] = (
                        diagonal if previous is None else previous + diagonal)
                    total_previous = head_total_diag[term]
                    head_total_diag[term] = (
                        diagonal if total_previous is None
                        else total_previous + diagonal)
                    head_diag[term][sector].block_until_ready()

            # Synchronize the small accumulator before advancing the block.
            # This is the lifetime boundary that prevents two W/G body tiles
            # from coexisting through asynchronous dispatch.
            sig_coh.block_until_ready()
            if verbose and jax.process_index() == 0:
                print_fn(f"  packed photon COHSEX block ({A},{B}) complete")

    from .cohsex_sigma import _replicate_band_sigma
    sig_x = _replicate_band_sigma(sig_x, mesh_xy)
    sig_sx = _replicate_band_sigma(sig_sx, mesh_xy)
    sig_coh = _replicate_band_sigma(sig_coh, mesh_xy)
    if wfns_charge.layout == "face":
        nb_sigma = wfns_charge.slices.nb_sigma
        sig_x = sig_x[:, :nb_sigma, :nb_sigma]
        sig_sx = sig_sx[:, :nb_sigma, :nb_sigma]
        sig_coh = sig_coh[:, :nb_sigma, :nb_sigma]
    sig_x.block_until_ready()
    sig_sx.block_until_ready()
    sig_coh.block_until_ready()
    if q0_factors is None:
        return sig_x, sig_sx, sig_coh, None
    diag_shape = (int(sig_x.shape[0]), int(sig_x.shape[1]))
    zero_diag = jnp.zeros(
        diag_shape, dtype=sig_x.dtype,
        device=NamedSharding(mesh_xy, P(None, None)))
    components = jnp.stack(
        [jnp.stack([
            zero_diag if head_diag[term][sector] is None
            else head_diag[term][sector]
            for sector in range(3)])
         for term in range(3)])
    components = device_put_process_local(
        components, NamedSharding(mesh_xy, P(None, None, None, None)))
    components.block_until_ready()
    direct_total = jnp.stack([
        jnp.zeros_like(zero_diag) if value is None else value
        for value in head_total_diag])
    sector_total = jnp.sum(components, axis=1)
    closure_abs = float(jax.device_get(jnp.max(jnp.abs(
        sector_total - direct_total))))
    closure_scale = float(jax.device_get(jnp.max(jnp.abs(direct_total))))
    closure_limit = 1.0e-13 + 1.0e-11 * closure_scale
    if closure_abs > closure_limit:
        raise ValueError(
            "GATE photon_head_sigma_sector_closure: final-block "
            "CC + CTTC + TT contraction does not close to the direct "
            f"16-block head total: {closure_abs:.3e} Ry > "
            f"{closure_limit:.3e} Ry")
    return (
        sig_x, sig_sx, sig_coh,
        StaticPhotonHeadSigmaDiagnostics(
            components,
            closure_abs,
            head_completion.hamiltonian_config_operator_fingerprint,
            "dft",
        ),
    )
