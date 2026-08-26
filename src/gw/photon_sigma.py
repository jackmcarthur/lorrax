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

import jax
import jax.numpy as jnp
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P


_CHANNELS = range(4)
_TERM_X = 0
_TERM_SX = 1
_TERM_COH = 2
_photon_sigma_kernel_cache: dict[tuple[object, ...], object] = {}


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
    mesh_xy: Mesh, kgrid, nk_tot: int, wfns_left, wfns_right,
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
           endpoint.get("face_shape"), endpoint.get("right_face_shape"))
    if key in _photon_sigma_kernel_cache:
        return _photon_sigma_kernel_cache[key]

    from common.contract_bands import contract_bands_block_reshard
    from .cohsex_sigma import _make_static_convolution, _occ_diag_full
    from .greens_function_kernel import build_G

    convolve = _make_static_convolution(mesh_xy, kgrid, nk_tot)
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
                       Gij, W_AB, V_AB, term):
        s_left = wfns_left.slices
        s_right = wfns_right.slices

        if wfns_left.layout == "legacy":
            def occupied(interaction):
                G = build_G(
                    wfns_left_g.xn(s_left.sigma),
                    wfns_right_g.yr(s_right.sigma), Gij=Gij)
                O = convolve(G, interaction, 1.0)
                return project(
                    wfns_left.xr(s_left.sigma), O,
                    wfns_right.yn(s_right.sigma))

            def coh(_):
                G = build_G(
                    wfns_left_g.xn(s_left.sigma_sum),
                    wfns_right_g.yr(s_right.sigma_sum))
                O = convolve(G, W_AB - V_AB, -0.5)
                return project(
                    wfns_left.xr(s_left.sigma), O,
                    wfns_right.yn(s_right.sigma))
        else:
            occ = _occ_diag_full(
                Gij, s_left.nb_sigma, s_left.nb_full)
            ri_mask = wfns_left.band_mask(
                s_left.sigma_sum).astype(jnp.complex128)

            def occupied(interaction):
                G = build_G(
                    wfns_left_g.psi_mun, wfns_right_g.psi_nmu,
                    phases=occ, layout="face", gemm=g_plan)
                O = convolve(G, interaction, 1.0)
                return project(wfns_left.psi_nmu, O, wfns_right.psi_mun)

            def coh(_):
                G = build_G(
                    wfns_left_g.psi_mun, wfns_right_g.psi_nmu,
                    phases=ri_mask, layout="face", gemm=g_plan)
                O = convolve(G, W_AB - V_AB, -0.5)
                return project(wfns_left.psi_nmu, O, wfns_right.psi_mun)

        def x_or_sx(_):
            interaction = jax.lax.cond(
                term == _TERM_X,
                lambda __: V_AB,
                lambda __: W_AB,
                operand=None,
            )
            return occupied(interaction)

        return jax.lax.cond(term == _TERM_COH, coh, x_or_sx, operand=None)

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
    print_fn=print,
    verbose: bool = True,
) -> tuple[jax.Array, jax.Array, jax.Array]:
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

    from .photon_layout import photon_block_view

    sig_x = None
    sig_sx = None
    sig_coh = None

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
                mesh_xy, meta.kgrid, int(meta.nk_tot), left, right)
            V_AB = photon_block_view(V_packed, photon_layout, A, B, mesh_xy)
            W_AB = photon_block_view(W_packed, photon_layout, A, B, mesh_xy)
            expected = (int(meta.nk_tot), n_left, n_right)
            if tuple(V_AB.shape) != expected or tuple(W_AB.shape) != expected:
                raise ValueError(
                    f"photon block ({A},{B}) shape mismatch: expected padded "
                    f"{expected} from its wavefunction endpoints, got "
                    f"V{tuple(V_AB.shape)} and W{tuple(W_AB.shape)}.")

            x_AB = contract_block(
                left, right, left_g, right_g, Gij, W_AB, V_AB,
                jnp.asarray(_TERM_X, dtype=jnp.int32))
            sig_x = x_AB if sig_x is None else sig_x + x_AB
            sig_x.block_until_ready()
            sx_AB = contract_block(
                left, right, left_g, right_g, Gij, W_AB, V_AB,
                jnp.asarray(_TERM_SX, dtype=jnp.int32))
            sig_sx = sx_AB if sig_sx is None else sig_sx + sx_AB
            sig_sx.block_until_ready()
            coh_AB = contract_block(
                left, right, left_g, right_g, Gij, W_AB, V_AB,
                jnp.asarray(_TERM_COH, dtype=jnp.int32))
            sig_coh = coh_AB if sig_coh is None else sig_coh + coh_AB

            # Synchronize the small accumulator before advancing the block.
            # This is the lifetime boundary that prevents two W/G body tiles
            # from coexisting through asynchronous dispatch.
            sig_coh.block_until_ready()
            if verbose and jax.process_index() == 0:
                print_fn(f"  full photon COHSEX block ({A},{B}) complete")

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
    return sig_x, sig_sx, sig_coh
