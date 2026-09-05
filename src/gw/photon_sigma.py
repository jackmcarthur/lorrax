"""Sixteen-block static COHSEX self-energy of ``bispinor_gw = full_static_cohsex``.

This is the Sigma owner of the packed static photon mode.  The screening
owner (:func:`gw.w_isdf.compute_static_photon_response`) stores the bare and
screened four-current propagators ``V^{AB}``, ``W^{AB}`` (Lorentz ``A,B in
{C, T1, T2, T3}``) as one packed, two-dimensionally sharded operator with
its Gamma cell already completed.  :func:`compute_static_photon_sigma`
streams the sixteen rectangular block views through the ordinary static
COHSEX kernels with Lorentz vertices applied to the Green operands,
accumulating ``Sigma_X`` (bare), ``Sigma_SX`` and ``Sigma_COH``
(screened) as one loop; ``Sigma^B`` (bare transverse exchange) is the TT
part of ``Sigma_X`` here, not a separate term.  The physics kernels remain
the existing owners:

* :func:`gw.greens_function_kernel.build_G` builds the rectangular Green
  function;
* :func:`gw.cohsex_sigma._make_static_convolution` performs the flat-k FFT
  convolution; and
* :func:`common.contract_bands.contract_bands_block_reshard` projects the
  exchange/correlation operator back to band space.

When the response carries a Gamma-cell completion
(:class:`gw.head_correction.StaticSlabPhotonHeadCompletion`, always under
``head_correction = full``), the same loop re-contracts the completion's
bounded rank-4 factors alone (``q0_only`` convolution,
:func:`gw.photon_layout.photon_q0_low_rank_block`) and reports the exact
diagonal contribution of the completed Gamma blocks per ``(X, SX, COH) x
(CC, CT+TC, TT)`` sector in :class:`StaticPhotonHeadSigmaDiagnostics`, gated
by ``GATE photon_head_sigma_sector_closure`` (the three sectors must sum to
the direct sixteen-block head total).  These diagnostics are what
``gw.gw_output.write_freq_debug`` prints as the ``*_CC/_CTTC/_TT`` columns.

This is ALSO the Sigma owner of the dynamic packed route (phase 3,
``compute_mode`` in the plasmon-pole pair with ``bispinor = true``).  There
the charge block's frequency dependence is carried by the ordinary scalar
Sigma_c machinery (:mod:`gw.ppm_pipeline` on the same ISDF ``W_00``) and the
current blocks are frozen at ``omega = 0``, so this function is called with
``blocks = "current"`` and contracts the twelve non-CC blocks only.  The
``(0,0)`` block is then skipped in the loop below and nowhere else: there is
one consumer, one head-sector closure gate and one set of kernels for both
routes.  See ``docs/theory/four-current-head-corrections.md`` and
``gw.sigma_dispatch.compute_sigma_xc``.

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
#: ``blocks`` selections of :func:`compute_static_photon_sigma`.  ``all`` is
#: the sixteen-block static COHSEX Sigma; ``current`` is the twelve non-CC
#: blocks, used by the dynamic packed route whose CC block is owned by the
#: scalar Sigma_c machinery at every frequency.
PHOTON_BLOCKS_ALL = "all"
PHOTON_BLOCKS_CURRENT = "current"
_PHOTON_BLOCK_SELECTIONS = (PHOTON_BLOCKS_ALL, PHOTON_BLOCKS_CURRENT)
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
    output_basis: str

    def __post_init__(self) -> None:
        shape = tuple(self.components_tskn_ry.shape)
        if len(shape) != 4 or shape[:2] != (3, 3):
            raise ValueError(
                "static photon head diagnostics must be (3 terms,3 sectors,"
                f"nk,nb); got {shape}")
        if self.output_basis != "dft":
            raise ValueError(
                "static photon head Sigma diagnostics must be stamped in "
                f"the DFT output basis; got {self.output_basis!r}")


@dataclass(frozen=True)
class StaticPhotonSigmaDiagnostics:
    """Lorentz-sector split of the physical static self-energy.

    ``components_skij_ry`` has sector order ``(CC, CT+TC, TT)`` and contains
    ``Sigma_SX + Sigma_COH`` for the selected blocks.  Unlike the head-only
    diagnostic above, these are band-space operators so the self-consistent
    driver can rotate them with the total Sigma before taking diagonals.
    """

    components_skij_ry: jax.Array
    max_closure_residual_ry: float

    def __post_init__(self) -> None:
        shape = tuple(self.components_skij_ry.shape)
        if len(shape) != 4 or shape[0] != 3 or shape[-2] != shape[-1]:
            raise ValueError(
                "static photon Sigma diagnostics must be "
                f"(3 sectors,nk,nb,nb); got {shape}")


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
    with_q0_diagnostic: bool, vertex_pair=(0, 0),
):
    """Contract X, SX or COH with vertices applied only to the Green operands."""
    from ffi import ffi_dial_key

    from .wavefunction_bundle import face_kernel_kwargs

    endpoint = face_kernel_kwargs(wfns_left, wfns_right)
    key = (id(mesh_xy), tuple(int(v) for v in kgrid), int(nk_tot),
           ffi_dial_key(), wfns_left.layout,
           endpoint.get("face_shape"), endpoint.get("right_face_shape"),
           bool(with_q0_diagnostic), tuple(vertex_pair))
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

    from common.gamma_matrices import gamma_apply, gamma_perm_phase

    @jax.jit
    def contract_block(wfns_left, wfns_right, Gij, W_AB, V_AB,
                       W_head_AB, V_head_AB, term):
        s = wfns_left.slices
        if wfns_left.layout == "face":
            direct, conjugated = wfns_left.psi_mun, wfns_right.psi_nmu
            bra, ket = wfns_left.psi_nmu, wfns_right.psi_mun
            occ = _occ_diag_full(Gij, s.nb_sigma, s.nb_full)
            ri = wfns_left.band_mask(s.sigma_sum).astype(jnp.complex128)
        else:
            direct, conjugated = wfns_left.psi_xn, wfns_right.psi_yr
            bra, ket = wfns_left.xr(s.sigma), wfns_right.yn(s.sigma)
        A, B = vertex_pair
        if A:
            direct = gamma_apply(direct, *gamma_perm_phase(A), axis=1)
        if B:
            conjugated = gamma_apply(conjugated, *gamma_perm_phase(B), axis=2)

        def green(is_coh):
            if wfns_left.layout == "face":
                phases = ri if is_coh else occ
                return build_G(direct, conjugated, phases=phases,
                               layout="face", gemm=g_plan)
            bands = s.sigma_sum if is_coh else s.sigma
            return build_G(direct[..., bands], conjugated[:, bands],
                           Gij=None if is_coh else Gij)

        def evaluate(G, interaction, head_interaction, sign):
            result = project(bra, convolve(G, interaction, sign), ket)
            head = (project(bra, convolve_q0(G, head_interaction, sign), ket)
                    if with_q0_diagnostic else result)
            return result, head

        def coh(_):
            return evaluate(green(True), W_AB - V_AB,
                            W_head_AB - V_head_AB, -0.5)

        def occupied(_):
            interaction, head = jax.lax.cond(
                term == _TERM_X, lambda _: (V_AB, V_head_AB),
                lambda _: (W_AB, W_head_AB), operand=None)
            return evaluate(green(False), interaction, head, 1.0)

        result, head = jax.lax.cond(
            term == _TERM_COH, coh, occupied, operand=None)
        return (result, head) if with_q0_diagnostic else result

    _photon_sigma_kernel_cache[key] = contract_block
    return contract_block


def contract_lorentz_blocks(
    blocks, *, carrier_C, carrier_T, plan_C, plan_T, term, mesh_xy,
    meta, Gij, get_block, with_q0_diagnostic=False,
    diagnostic_basis_rotation=None, print_fn=print, verbose=True,
):
    """Sum requested Lorentz blocks, synchronizing each tile before restoring the next."""
    from .wavefunction_bundle import padded_centroid_extent

    if plan_C is not None or plan_T is not None:
        raise NotImplementedError("Lorentz parent contraction awaits the S3 unfold seam")
    sig = [None, None, None]
    sectors = [None, None, None]
    head_diag = [[None] * 3 for _ in range(3)]
    head_total = [None, None, None]
    for A, B in blocks:
        left = _bundle_for_channel(carrier_C, carrier_T, A)
        right = _bundle_for_channel(carrier_C, carrier_T, B)
        kernel = _make_photon_static_block_kernel(
            mesh_xy, meta.kgrid, int(meta.nk_tot), left, right,
            with_q0_diagnostic=with_q0_diagnostic, vertex_pair=(A, B))
        V, W, V_head, W_head = get_block(A, B)
        expected = (int(meta.nk_tot), padded_centroid_extent(left),
                    padded_centroid_extent(right))
        if V.shape != expected or W.shape != expected:
            raise ValueError(f"photon block ({A},{B}): expected {expected}, "
                             f"got V{V.shape} and W{W.shape}")
        sector = _head_sector(A, B)
        physical = None
        for t in term:
            result = kernel(left, right, Gij, W, V, W_head, V_head,
                            jnp.asarray(t, dtype=jnp.int32))
            value, head = result if with_q0_diagnostic else (result, None)
            sig[t] = value if sig[t] is None else sig[t] + value
            sig[t].block_until_ready()
            if t in (_TERM_SX, _TERM_COH):
                physical = value if physical is None else physical + value
            if with_q0_diagnostic:
                diagonal = _diagnostic_diagonal(
                    _sigma_window_matrix(head, carrier_C),
                    diagnostic_basis_rotation, mesh_xy)
                previous = head_diag[t][sector]
                head_diag[t][sector] = diagonal if previous is None else previous + diagonal
                previous = head_total[t]
                head_total[t] = diagonal if previous is None else previous + diagonal
                head_diag[t][sector].block_until_ready()
        if physical is not None:
            previous = sectors[sector]
            sectors[sector] = physical if previous is None else previous + physical
            sectors[sector].block_until_ready()
        del V, W, V_head, W_head, result, value, head, physical
        if verbose and jax.process_index() == 0:
            print_fn(f"  photon Sigma block ({A},{B}) complete")
    return sig, sectors, head_diag, head_total


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
    blocks: str = PHOTON_BLOCKS_ALL,
    head_completion=None,
    diagnostic_basis_rotation=None,
    diagnostic_input_basis=None,
    print_fn=print,
    verbose: bool = True,
) -> tuple[
    jax.Array, jax.Array, jax.Array, StaticPhotonHeadSigmaDiagnostics | None,
    StaticPhotonSigmaDiagnostics,
]:
    """Stream the ``D^{AB}`` blocks into full static COHSEX.

    ``V_packed`` and ``W_packed`` must stay at ``P(None, 'x', 'y')``.
    :func:`gw.photon_layout.photon_block_view` returns a mesh-aligned padded
    view, whose two extents must equal those of the corresponding charge or
    transverse wavefunction bundle.  No logical block is copied or gathered.

    ``blocks`` selects which of the sixteen are summed, and is the ONLY
    difference between the static and the dynamic packed route's use of this
    function:

    * ``"all"`` (the default, ``compute_mode = cohsex``) -- all sixteen.
    * ``"current"`` (the dynamic packed route) -- the twelve blocks with at
      least one current index.  The ``(0,0)`` block is skipped because the
      dynamic route's charge channel is owned end to end by the scalar
      ``Sigma_x + Sigma_c(omega)`` machinery on the same ``W_00``; summing it
      here as well would double count it, statically.

    The head-sector closure gate is unaffected by the selection: the skipped
    blocks are absent from the direct total and from the per-sector sums
    alike, so the CC sector is exactly zero under ``"current"`` rather than
    partially populated.
    """
    if blocks not in _PHOTON_BLOCK_SELECTIONS:
        raise ValueError(
            f"photon Sigma block selection must be one of "
            f"{_PHOTON_BLOCK_SELECTIONS}; got {blocks!r}")
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

    def get_block(A, B):
        V = photon_block_view(V_packed, photon_layout, A, B, mesh_xy)
        W = photon_block_view(W_packed, photon_layout, A, B, mesh_xy)
        if q0_factors is None:
            return V, W, V, W
        return (V, W, photon_q0_low_rank_block(
            (q0_factors.bare_pair,), photon_layout, A, B, mesh_xy),
            photon_q0_low_rank_block(
                q0_factors.screened_pairs, photon_layout, A, B, mesh_xy))

    selected = [(A, B) for A in _CHANNELS for B in _CHANNELS
                if blocks != PHOTON_BLOCKS_CURRENT or (A, B) != (0, 0)]
    sig, sigma_sector, head_diag, head_total_diag = contract_lorentz_blocks(
        selected, carrier_C=wfns_charge, carrier_T=wfns_transverse,
        plan_C=None, plan_T=None, term=(_TERM_X, _TERM_SX, _TERM_COH),
        mesh_xy=mesh_xy, meta=meta, Gij=Gij, get_block=get_block,
        with_q0_diagnostic=q0_factors is not None,
        diagnostic_basis_rotation=diagnostic_basis_rotation,
        print_fn=print_fn, verbose=verbose)
    sig_x, sig_sx, sig_coh = sig

    from .cohsex_sigma import _replicate_band_sigma
    sig_x = _replicate_band_sigma(sig_x, mesh_xy)
    sig_sx = _replicate_band_sigma(sig_sx, mesh_xy)
    sig_coh = _replicate_band_sigma(sig_coh, mesh_xy)
    sigma_sector = [
        None if value is None else _replicate_band_sigma(value, mesh_xy)
        for value in sigma_sector]
    if wfns_charge.layout == "face":
        nb_sigma = wfns_charge.slices.nb_sigma
        sig_x = sig_x[:, :nb_sigma, :nb_sigma]
        sig_sx = sig_sx[:, :nb_sigma, :nb_sigma]
        sig_coh = sig_coh[:, :nb_sigma, :nb_sigma]
        sigma_sector = [
            None if value is None
            else value[:, :nb_sigma, :nb_sigma]
            for value in sigma_sector]
    sig_x.block_until_ready()
    sig_sx.block_until_ready()
    sig_coh.block_until_ready()
    zero_matrix = jnp.zeros_like(sig_sx)
    sigma_components = jnp.stack([
        zero_matrix if value is None else value for value in sigma_sector])
    sigma_components = device_put_process_local(
        sigma_components, NamedSharding(mesh_xy, P(None, None, None, None)))
    sigma_components.block_until_ready()
    sigma_closure_abs = float(jax.device_get(jnp.max(jnp.abs(
        jnp.sum(sigma_components, axis=0) - (sig_sx + sig_coh)))))
    sigma_closure_scale = float(jax.device_get(jnp.max(jnp.abs(
        sig_sx + sig_coh))))
    sigma_closure_limit = 1.0e-13 + 1.0e-11 * sigma_closure_scale
    if sigma_closure_abs > sigma_closure_limit:
        raise ValueError(
            "GATE photon_sigma_sector_closure: CC + CTTC + TT does not "
            f"close to Sigma_SX + Sigma_COH: {sigma_closure_abs:.3e} Ry > "
            f"{sigma_closure_limit:.3e} Ry")
    sigma_diagnostics = StaticPhotonSigmaDiagnostics(
        sigma_components, sigma_closure_abs)
    if q0_factors is None:
        return sig_x, sig_sx, sig_coh, None, sigma_diagnostics
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
            "GATE photon_head_sigma_sector_closure: projected photon-head "
            "sectors do not close to the direct contraction.\n"
            f"  got:  closure_abs = {closure_abs:.3e} Ry\n"
            f"  want: closure_abs <= {closure_limit:.3e} Ry\n"
            "  why:  a CC + CTTC + TT sum that differs from the direct "
            "16-block total has lost or double-counted a Lorentz sector")
    return (
        sig_x, sig_sx, sig_coh,
        StaticPhotonHeadSigmaDiagnostics(components, closure_abs, "dft"),
        sigma_diagnostics,
    )
