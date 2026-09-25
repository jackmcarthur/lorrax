"""Inputs of the packed static photon Gamma-cell completion.

``bispinor_gw = full_static_cohsex`` completes the ``q = Gamma, G = 0`` slot
of the packed sixteen-block V and W (``gw.head_correction.
complete_static_slab_photon_q0``) from one bounded response

    ``R(q) = q_a H_a(sigma_H) + q_a q_b S_ab``

that this module assembles.  Its content, by declaration:

* present: the charge CC ``q^2`` head ``S^{00}`` from the incumbent scalar
  producer (:func:`gw.qsgw_head.build_dft_head_response` at ``omega = 0``),
  the charge one-leg wings ``Y^{0}``/``Z^{0}`` that fold it through the
  packed body, and the Hall CT/TC ``q^1`` term generated structurally from
  ``sigma_H`` (:func:`gw.head_correction.static_hall_linear_response`);
* omitted by model: the current ``q^2`` response (TT, CT/TC), the current
  wings, the uniform static current response ``tt_q0`` (zero by gauge
  invariance for an insulator), the diamagnetic/contact terms and the
  negative-energy (complement-space) closure.  They are never stored as
  accidental zeros of a larger schema; ``S_direct`` has charge support only.

The Hall term is optional.  ``sigma_H`` comes from the immutable artifact
written by ``get_dipole_mtxels --static-gauge-hall-only`` when the deck's
``static_gauge_hall_file`` exists and authenticates against the run's WFN,
band manifold and k-count; when the file is absent ``sigma_H = 0`` and
``hall_source`` says so.  For a Chern-trivial insulator the static Hall
coefficient is exactly zero in the converged limit (it is the occupied
Berry-curvature sum, i.e. a Chern number), so the absent-artifact default is
the exact answer for the systems this mode admits.

This module owns no WFN load, symmetry unfold, current, FFT, body-response
or packing implementation.  It composes the existing routines and retains
only O(N_mu) wing carriers.  The record it issues is sealed: only the
producer below can construct it, so a fabricated response cannot reach the
completion.
"""
from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

from common.collectives import device_put_process_local
from gw.photon_layout import PhotonBasisLayout, pack_photon_channel_vectors


_PRODUCER_TOKEN = object()
HALL_SOURCE_NONE = "none: sigma_H = 0 (no static_gauge_hall_file)"


def fermi_dirac_current_drude(current_faces, occupation_state, *,
                              state_capacity, cell_volume, kweights, mesh_xy):
    r"""Return centroid-resolved D from the bank's diagonal current densities.

    Implements ``D_mu,nu = C/Omega sum_kn w_k (-f'_kn)
    j_kn(mu) conj(j_kn(nu))``, with ``-f'=f(1-f)/kT``. The packed
    centroid axis includes the current component. Charge slots must be zero.
    This body contact is built once per bank; it is not a head update.

    Parameters
    ----------
    current_faces : tuple of jax.Array
        Diagonal transition densities at the same endpoints as the stream:
        ``[k,mu,band]`` at ``P(None,'x','y')`` and ``[k,band,nu]`` at
        ``P(None,'x','y')``. Only linear-size carriers redistribute.
    occupation_state : OccupationState
        The bank's FD occupations and smearing width in Ry.
    state_capacity : float
        Physical occupancy per normalized state from the WFN loader.
    cell_volume : float
        Cell volume in bohr cubed.
    kweights : array_like
        Normalized full-BZ weights ``[k]``; uniform grid uses ``1/Nk``.
    mesh_xy : Mesh
        Named x/y mesh. The result is complex ``[mu,nu]`` at ``P('x','y')``.
    """
    from common.shard_map import shard_map

    if occupation_state.smearing_family != "fd":
        raise ValueError("GATE photon_contact_fd: TT metal contact requires FD state")
    width = float(occupation_state.smearing_width_ry)
    capacity, volume = float(state_capacity), float(cell_volume)
    left, right = current_faces
    f = jnp.asarray(occupation_state.f_kn)
    weights = np.asarray(kweights, dtype=np.float64)
    if (left.ndim != 3 or right.ndim != 3
            or (left.shape[0], left.shape[2]) != f.shape
            or right.shape[:2] != f.shape):
        raise ValueError("GATE photon_contact_current: want diagonal current faces [k,mu,n], [k,n,nu]")
    if (not np.isfinite([width, capacity, volume]).all()
            or min(width, capacity, volume) <= 0
            or weights.shape != (f.shape[0],) or not np.isfinite(weights).all()
            or np.any(weights < 0) or not np.isclose(weights.sum(), 1, rtol=0, atol=1e-12)):
        raise ValueError("GATE photon_contact_state: invalid FD scale, volume or k weights")
    # Replicate only the band axis of these O(k*n*mu) carriers. Every
    # quadratic output remains tiled over BOTH mesh axes throughout.
    contract = shard_map(
        lambda l, r, w: jnp.einsum("kmn,kn,knv->mv", l, w, r.conj()),
        mesh=mesh_xy, in_specs=(P(None, "x", None), P(None, None, "y"), P()),
        out_specs=P("x", "y"), check_vma=False)
    weight = (capacity / volume) * jnp.asarray(weights)[:, None] * f * (1-f) / width
    return jax.jit(contract)(left, right, weight)


def photon_diagonal_current_faces(vertex, *, mesh_xy, layout, wfn_layout="face"):
    """Contract psi-dagger J psi at each centroid of the stream endpoints, at full k.

    ``vertex`` is ``prepare_photon_carriers``'s endpoints.  The current
    density ``j^A = psi^dagger alpha_A psi`` is formed on the current
    family's raw parents and transported to full k as the polar time-odd
    vector it is: the typed action's centroid pullback with
    ``SymMaps.cartesian_action`` in place of the spin action and no Bloch
    phase (``symmetry_maps.unfold_wavefunction_local``); no psi face is
    unfolded.  Returned in the canonical ``layout``: ``[k, mu, band]`` and
    ``[k, band, nu]``; the charge channel and internal padding are exactly
    zero, so their outer product has TT support only.
    """
    from functools import partial
    from common.gamma_matrices import gamma_apply, gamma_perm_phase
    from common.shard_map import shard_map
    from common.wfn_layout import psi_specs
    from gw.photon_layout import TRANSVERSE, pack_photon_faces
    from symmetry_maps import unfold_wavefunction_local

    families = vertex.families
    plan = families.plans[1]
    nmu_spec, mun_spec = psi_specs(wfn_layout)

    def density(face, spin_axis):
        return jnp.stack([jnp.sum(jnp.conj(face) * gamma_apply(
            face, *gamma_perm_phase(A), axis=spin_axis), axis=spin_axis)
            for A in TRANSVERSE], axis=spin_axis)

    tables = ()
    if plan is not None:
        rows = np.asarray(plan.sym_idx)
        tables = (np.asarray(plan.irr_idx), rows, np.asarray(plan.k_parent_frac),
                  np.asarray(plan.centroid_local_perm),
                  np.zeros(np.shape(plan.L_table), np.float64),
                  np.asarray(plan.sym.cartesian_action(rows, axial=False, time_odd=True),
                             dtype=np.complex128))

    def to_full_k(value, spin_axis, mu_axis, mesh_axis, *tables):
        if not tables:
            return value
        irr, sym, kfrac, perm, wraps, mix = tables
        return unfold_wavefunction_local(
            value, irr_idx=irr, sym_idx=sym, k_irr_frac=kfrac, local_perm=perm,
            L_table=wraps, spin_action_full=mix, n_sym_spatial=plan.n_sym_spatial,
            spin_axis=spin_axis, mu_axis=mu_axis, mesh_axis=mesh_axis)

    table_specs = (P(),) * len(tables)

    @partial(shard_map, mesh=mesh_xy, in_specs=(mun_spec, nmu_spec) + table_specs,
             out_specs=(mun_spec, nmu_spec), check_vma=False)
    def currents(mun, nmu, *tables):
        return (to_full_k(density(mun, 1), 1, 2, "x", *tables),
                to_full_k(density(nmu, 2), 2, 3, "y", *tables))

    left, right = jax.jit(currents)(vertex.mun[1], vertex.nmu[1], *tables)
    if families.bases is not None:
        basis = families.bases[1]
        left = basis.unpack_axis(left, 2, spec=mun_spec)
        right = basis.unpack_axis(right, 3, spec=nmu_spec)
    def zero(face, axis, spec):
        # The charge channel carries no current: an exact-zero face.
        shape = face.shape[:axis] + (1, layout.carrier_extent(0)) + face.shape[axis+2:]
        return jax.jit(lambda: jnp.zeros(shape, face.dtype),
                       out_shardings=NamedSharding(mesh_xy, spec))()
    faces_mun = (zero(left, 1, mun_spec),) + tuple(left[:, A:A+1] for A in range(3))
    faces_nmu = (zero(right, 2, nmu_spec),) + tuple(right[:, :, A:A+1] for A in range(3))
    left = pack_photon_faces(faces_mun, layout, mesh_xy, orientation="mun",
                             wfn_layout=wfn_layout)[:, 0]
    right = pack_photon_faces(faces_nmu, layout, mesh_xy, orientation="nmu",
                              wfn_layout=wfn_layout)[:, :, 0]
    return left, right


def _canonical_wfn_sha256(value) -> str:
    value = str(value).strip()
    if (len(value) != 64
            or any(c not in "0123456789abcdef" for c in value)):
        raise ValueError(
            "static photon head WFN identifier must be 64 lowercase hex")
    return value


@dataclass(frozen=True)
class StaticPhotonHeadResponse:
    r"""Inputs for ``R(q)=q_a H_a(sigma_H)+q_a q_b S_ab``.

    ``S_direct`` and the wings have charge support only.  The Hall tensor is
    intentionally not stored: :func:`gw.head_correction.
    static_hall_linear_response` constructs it from ``sigma_H`` at the
    numerical kernel that consumes it.  ``hall_source`` records where
    ``sigma_H`` came from (the authenticated artifact path, or
    :data:`HALL_SOURCE_NONE`).
    """

    layout: PhotonBasisLayout
    S_direct: jax.Array               # (2,2,4,4), replicated charge CC
    sigma_H: jax.Array                # (3,), replicated real bohr^-1
    hall_source: str
    Y_x: jax.Array                    # (2,4,Npacked), P(None,None,'x')
    Z_y: jax.Array                    # (2,Npacked,4), P(None,'y',None)
    ward_residual: float
    hermiticity_residual: float
    wing_reciprocity_residual: float
    _producer_token: object

    def __post_init__(self) -> None:
        if self._producer_token is not _PRODUCER_TOKEN:
            raise TypeError(
                "StaticPhotonHeadResponse is issued only by "
                "build_static_photon_head_response")


def _same_mesh_sharding(array, mesh: Mesh, spec: P) -> bool:
    sharding = getattr(array, "sharding", None)
    return (
        isinstance(sharding, NamedSharding)
        and tuple(sharding.mesh.axis_names) == tuple(mesh.axis_names)
        and np.array_equal(sharding.mesh.devices, mesh.devices)
        and sharding.is_equivalent_to(NamedSharding(mesh, spec), array.ndim)
    )


def require_static_photon_head_response(
    response: StaticPhotonHeadResponse, mesh_xy: Mesh,
) -> StaticPhotonHeadResponse:
    """Check the bounded model's support, dtype and sharding."""
    if not isinstance(response, StaticPhotonHeadResponse):
        raise TypeError(
            "the packed static photon completion requires "
            f"StaticPhotonHeadResponse; got {type(response).__name__}")
    response.layout.assert_mesh(mesh_xy)

    n_packed = int(response.layout.packed_extent)
    arrays = (
        (response.S_direct, "S_direct", (2, 2, 4, 4), np.complex128, P()),
        (response.sigma_H, "sigma_H", (3,), np.float64, P()),
        (response.Y_x, "Y_x", (2, 4, n_packed), np.complex128,
         P(None, None, "x")),
        (response.Z_y, "Z_y", (2, n_packed, 4), np.complex128,
         P(None, "y", None)),
    )
    for array, name, shape, dtype, spec in arrays:
        if tuple(array.shape) != shape:
            raise ValueError(f"{name} shape {array.shape} != {shape}")
        if np.dtype(array.dtype) != np.dtype(dtype):
            raise TypeError(f"{name} dtype {array.dtype} != {np.dtype(dtype)}")
        if not _same_mesh_sharding(array, mesh_xy, spec):
            raise ValueError(f"{name} must have production sharding {spec}")

    S = np.asarray(jax.device_get(response.S_direct))
    sigma = np.asarray(jax.device_get(response.sigma_H))
    if not np.all(np.isfinite(sigma)):
        raise ValueError("static photon head sigma_H is not finite")
    if np.any(S[:, :, 0, 1:] != 0.0) or np.any(S[:, :, 1:, :] != 0.0):
        raise ValueError("static photon head S has non-charge support")
    from gw.head_correction import static_gauge_tensor_residuals
    structural_ward, structural_hermiticity = (
        static_gauge_tensor_residuals(S))
    residuals = np.asarray((
        response.ward_residual, response.hermiticity_residual,
        response.wing_reciprocity_residual,
        structural_ward, structural_hermiticity), dtype=np.float64)
    if np.any(~np.isfinite(residuals)) or np.any(residuals < 0.0):
        raise ValueError("static photon head response has invalid residuals")
    if max(response.ward_residual, structural_ward) > 1.0e-8:
        raise ValueError(
            "static photon head response violates the static Ward gate")
    if max(response.hermiticity_residual,
           structural_hermiticity) > 1.0e-10:
        raise ValueError(
            "static photon head response violates the Hermiticity gate")
    if response.wing_reciprocity_residual > 1.0e-10:
        raise ValueError(
            "static photon head response violates wing reciprocity")
    return response


def _replicated(value, mesh: Mesh, *, dtype):
    return device_put_process_local(
        np.asarray(value, dtype=dtype), NamedSharding(mesh, P()))


def _channel_zeros(nq: int, extent: int, mesh: Mesh, axis: str):
    return device_put_process_local(
        np.zeros((int(nq), int(extent)), dtype=np.complex128),
        NamedSharding(mesh, P(None, axis)))


def build_static_photon_head_response(
    wfns,
    *,
    input_dir: str,
    mesh: Mesh,
    wfn,
    meta,
    config,
    layout: PhotonBasisLayout,
    hall_transaction=None,
    wfn_fingerprint_binding=None,
) -> StaticPhotonHeadResponse:
    r"""Compose the charge CC head, its wings and the optional Hall term.

    The scalar response is evaluated once by
    :func:`gw.qsgw_head.build_dft_head_response` at ``omega = 0``.  Its two
    in-plane velocity rows become the qx/qy derivatives of the charge head
    and charge-only wings.  Three exact-zero transverse vectors are passed to
    the canonical packer; no current wing is inferred.

    ``hall_transaction`` is either ``None`` (``sigma_H = 0``) or the full-BZ
    result of :func:`gw.qsgw_head.static_gauge_hall_transaction`, which must
    name the same WFN identity and band manifold as the charge response.
    """
    from common.parallel_transport import (
        fingerprint_from_binding, wfn_fingerprint)
    from gw.head_correction import (
        canonicalize_static_gauge_q2_tensor,
        static_gauge_tensor_residuals,
    )
    from gw.qsgw_head import (
        StaticGaugeHallTransaction, build_dft_head_response)

    if not isinstance(layout, PhotonBasisLayout):
        raise TypeError(
            "static photon head response requires PhotonBasisLayout")
    layout.assert_mesh(mesh)

    wfn_fp = _canonical_wfn_sha256(
        wfn_fingerprint(wfn)
        if wfn_fingerprint_binding is None
        else fingerprint_from_binding(wfn_fingerprint_binding, wfn))
    start, stop = int(meta.b_id_0), int(meta.b_id_4_chi_user)
    if hall_transaction is None:
        sigma_host = np.zeros(3, dtype=np.float64)
        hall_source = HALL_SOURCE_NONE
    else:
        if not isinstance(hall_transaction, StaticGaugeHallTransaction):
            raise TypeError(
                "static photon head response requires the sealed full-BZ "
                "Hall transaction or None")
        if (int(hall_transaction.band_start),
                int(hall_transaction.band_stop)) != (start, stop):
            raise ValueError(
                "charge and Hall responses use different band manifolds: "
                f"charge=[{start},{stop}), Hall=[{hall_transaction.band_start},"
                f"{hall_transaction.band_stop})")
        if _canonical_wfn_sha256(hall_transaction.wfn_fingerprint) != wfn_fp:
            raise ValueError(
                "charge and Hall responses use different WFN identities")
        sigma_host = np.asarray(
            jax.device_get(hall_transaction.sigma_H), dtype=np.float64)
        if sigma_host.shape != (3,) or not np.all(np.isfinite(sigma_host)):
            raise ValueError("Hall transaction has an invalid sigma_H")
        hall_source = (
            f"{hall_transaction.producer_id} "
            f"(operator {hall_transaction.hamiltonian_config_operator_fingerprint})")

    direct = build_dft_head_response(
        wfns, (0.0 + 0.0j,), input_dir=input_dir, mesh=mesh,
        wfn=wfn, meta=meta, config=config,
        wfn_fingerprint_binding=wfn_fingerprint_binding)
    if direct.Y_x is None or direct.Z_y is None:
        raise ValueError("incumbent charge response returned no body wings")
    charge_extent = int(layout.carrier_extent(0))
    if (tuple(direct.Y_x.shape) != (1, 3, charge_extent)
            or tuple(direct.Z_y.shape) != (1, charge_extent, 3)):
        raise ValueError(
            "charge response/layout mismatch: "
            f"Y={direct.Y_x.shape}, Z={direct.Z_y.shape}, "
            f"charge padded extent={charge_extent}")

    charge_y = direct.Y_x[0, :2, :]
    charge_z = jnp.transpose(direct.Z_y[0, :, :2], (1, 0))
    zeros_x = tuple(
        _channel_zeros(2, layout.carrier_extent(A), mesh, "x")
        for A in range(1, 4))
    zeros_y = tuple(
        _channel_zeros(2, layout.carrier_extent(A), mesh, "y")
        for A in range(1, 4))
    Y_x = pack_photon_channel_vectors(
        (charge_y, *zeros_x), layout, mesh, axis_name="x")
    Z_packed_y = pack_photon_channel_vectors(
        (charge_z, *zeros_y), layout, mesh, axis_name="y")
    Z_y = jnp.transpose(Z_packed_y, (0, 2, 1))

    S_host = np.zeros((2, 2, 4, 4), dtype=np.complex128)
    charge_S = np.asarray(
        jax.device_get(direct.S_direct[0, :2, :2]), dtype=np.complex128)
    S_host[:, :, 0, 0] = charge_S
    S_direct = canonicalize_static_gauge_q2_tensor(
        _replicated(S_host, mesh, dtype=np.complex128))
    sigma_H = _replicated(sigma_host, mesh, dtype=np.float64)
    # At static imaginary frequency the Adler--Wiser weight is real, hence
    # the two incumbent wing orientations obey Z[b,mu]=conj(Y[b,mu]).  Move
    # only this O(N_mu) vector to Y sharding for a scalar certificate.
    charge_z_x = device_put_process_local(
        charge_z, NamedSharding(mesh, P(None, "x")))
    wing_delta = jnp.max(jnp.abs(charge_y - jnp.conj(charge_z_x)))
    wing_scale = jnp.maximum(
        jnp.maximum(jnp.max(jnp.abs(charge_y)),
                    jnp.max(jnp.abs(charge_z_x))), 1.0e-300)
    wing_reciprocity = float(
        np.asarray(jax.device_get(wing_delta / wing_scale)))
    ward, hermiticity = static_gauge_tensor_residuals(S_direct)

    response = StaticPhotonHeadResponse(
        layout=layout,
        S_direct=S_direct, sigma_H=sigma_H, hall_source=hall_source,
        Y_x=Y_x, Z_y=Z_y,
        ward_residual=float(ward), hermiticity_residual=float(hermiticity),
        wing_reciprocity_residual=wing_reciprocity,
        _producer_token=_PRODUCER_TOKEN,
    )
    return require_static_photon_head_response(response, mesh)


__all__ = [
    "HALL_SOURCE_NONE",
    "StaticPhotonHeadResponse",
    "build_static_photon_head_response",
    "require_static_photon_head_response",
]
