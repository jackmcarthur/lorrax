"""Inputs for the bounded static charge-plus-Hall response.

``charge_hall_cubature`` is a deliberately restricted effective response:
the existing scalar-density head/wings at omega=0 plus the separately computed
Hall Chern--Simons term.  Ordinary current response,
contact and complement-space closure are omitted *by model*, never stored as
accidental zeros and never promoted to ``full_static_gauge``.

This module also composes the retained width-eight first jet with the
authenticated charge/transverse centroid bases to issue the local-alpha Y/Z
first-order one-leg algebra carriers.  That object is deliberately not a
response: exact gauged-VNL current wings, weight/state second jets, contact,
and complement closure remain absent.

This module owns no WFN load, symmetry unfold, current, FFT, body-response or
packing implementation.  It composes the existing routines and retains only
O(N_mu) wing carriers.
"""
from __future__ import annotations

from dataclasses import dataclass, fields
import enum

import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

from gw.photon_layout import (
    PhotonBasisLayout,
    pack_photon_channel_vectors,
    pack_photon_head_body_vectors,
)


class StaticGaugeResponseCapability(str, enum.Enum):
    CHARGE_HALL_CUBATURE = "charge_hall_cubature"
    FULL_STATIC_GAUGE = "full_static_gauge"


class StaticGaugeTermStatus(str, enum.Enum):
    COMPLETE = "complete"
    OMITTED_BY_MODEL = "omitted_by_model"
    UNAVAILABLE = "unavailable"


@dataclass(frozen=True)
class StaticGaugeTermAvailability:
    """Fixed response-jet availability grammar; fields are never inferred."""

    cc_q2: StaticGaugeTermStatus
    ct_q1: StaticGaugeTermStatus
    tc_q1: StaticGaugeTermStatus
    ct_q2: StaticGaugeTermStatus
    tc_q2: StaticGaugeTermStatus
    tt_q0: StaticGaugeTermStatus
    tt_q1: StaticGaugeTermStatus
    tt_q2: StaticGaugeTermStatus
    y_charge: StaticGaugeTermStatus
    y_current: StaticGaugeTermStatus
    z_charge: StaticGaugeTermStatus
    z_current: StaticGaugeTermStatus
    contact_q0: StaticGaugeTermStatus
    contact_q2: StaticGaugeTermStatus
    complement_space: StaticGaugeTermStatus

    def as_tokens(self) -> dict[str, str]:
        return {field.name: getattr(self, field.name).value
                for field in fields(self)}

    def is_complete_for(self, capability: StaticGaugeResponseCapability) -> bool:
        capability = StaticGaugeResponseCapability(capability)
        status = self.as_tokens()
        if capability is StaticGaugeResponseCapability.FULL_STATIC_GAUGE:
            return all(value == StaticGaugeTermStatus.COMPLETE.value
                       for value in status.values())
        required = {"cc_q2", "ct_q1", "tc_q1", "y_charge", "z_charge"}
        return all(
            (value == StaticGaugeTermStatus.COMPLETE.value) == (name in required)
            and (name in required
                 or value == StaticGaugeTermStatus.OMITTED_BY_MODEL.value)
            for name, value in status.items())

    def require_for(self, capability: StaticGaugeResponseCapability) -> None:
        capability = StaticGaugeResponseCapability(capability)
        if not self.is_complete_for(capability):
            raise ValueError(
                f"GATE static_gauge_availability: status {self.as_tokens()} "
                f"does not satisfy capability {capability.value!r}")


def _charge_hall_availability() -> StaticGaugeTermAvailability:
    complete = StaticGaugeTermStatus.COMPLETE
    omitted = StaticGaugeTermStatus.OMITTED_BY_MODEL
    return StaticGaugeTermAvailability(
        cc_q2=complete, ct_q1=complete, tc_q1=complete,
        ct_q2=omitted, tc_q2=omitted,
        tt_q0=omitted, tt_q1=omitted, tt_q2=omitted,
        y_charge=complete, y_current=omitted,
        z_charge=complete, z_current=omitted,
        contact_q0=omitted, contact_q2=omitted,
        complement_space=omitted,
    )


CHARGE_HALL_CUBATURE_AVAILABILITY = _charge_hall_availability()
_PRODUCER_TOKEN = object()
_RETAINED_ALPHA_HEAD_WING_TOKEN = object()


def _canonical_wfn_sha256(value) -> str:
    value = str(value).strip()
    if (len(value) != 64
            or any(c not in "0123456789abcdef" for c in value)):
        raise ValueError(
            "charge/Hall WFN identifier must be 64 lowercase hex")
    return value


@dataclass(frozen=True)
class ChargeHallCubatureResponse:
    r"""Inputs for ``R(q)=q_a H_a(sigma_H)+q_a q_b S_ab``.

    ``S_direct`` and the wings have charge support only.  The Hall tensor is
    intentionally not stored: :func:`static_hall_linear_response` constructs
    it from ``sigma_H`` at the numerical kernel that consumes it.
    """

    capability: StaticGaugeResponseCapability
    availability: StaticGaugeTermAvailability
    layout: PhotonBasisLayout
    S_direct: jax.Array               # (2,2,4,4), replicated charge CC
    sigma_H: jax.Array                # (3,), replicated real bohr^-1
    Y_x: jax.Array                    # (2,4,Npacked), P(None,None,'x')
    Z_y: jax.Array                    # (2,Npacked,4), P(None,'y',None)
    ward_residual: float
    hermiticity_residual: float
    wing_reciprocity_residual: float
    _producer_token: object

    def __post_init__(self) -> None:
        if self._producer_token is not _PRODUCER_TOKEN:
            raise TypeError(
                "ChargeHallCubatureResponse is issued only by "
                "build_charge_hall_cubature_response")


@dataclass(frozen=True)
class RetainedAlphaFirstOrderHeadWingCarriers:
    r"""Basis-authenticated ``D1(head) * alpha(body)`` algebra carriers.

    ``Y_x[omega,a,I,body]`` and ``Z_y[omega,a,body,I]`` contain all four
    body Lorentz channels in the canonical packed photon basis.  The charge
    body channel uses the authenticated charge centroid basis; T1/T2/T3 use
    the one authenticated transverse basis and canonical local alpha-current
    insertion inside :func:`gw.qsgw_head.head_wings_sharded`.

    This is only retained-bubble algebra produced by the in-memory width-eight
    energy-scaled jet.  The receipts authenticate the two centroid bases to
    one WFN manifold; they do not authenticate that jet as a persisted
    same-Hamiltonian artifact.  The exact finite-transfer gauged-VNL body
    current, current-wing reciprocity/symmetry action, response-weight/state
    derivatives, contact, and complement closure are absent.  Consequently
    this is not a :class:`ChargeHallCubatureResponse`, carries no capability,
    and does not change :class:`StaticGaugeTermAvailability`.
    """

    layout: PhotonBasisLayout
    Y_x: jax.Array                 # (nw,2,4,Npacked), last axis x-sharded
    Z_y: jax.Array                 # (nw,2,Npacked,4), body axis y-sharded
    charge_basis_receipt: object
    transverse_basis_receipt: object
    nb_logical: int
    n_omega: int
    _producer_token: object

    def __post_init__(self) -> None:
        if self._producer_token is not _RETAINED_ALPHA_HEAD_WING_TOKEN:
            raise TypeError(
                "RetainedAlphaFirstOrderHeadWingCarriers is issued only by "
                "build_retained_alpha_first_order_head_wings")


def _same_mesh_sharding(array, mesh: Mesh, spec: P) -> bool:
    sharding = getattr(array, "sharding", None)
    return (
        isinstance(sharding, NamedSharding)
        and tuple(sharding.mesh.axis_names) == tuple(mesh.axis_names)
        and np.array_equal(sharding.mesh.devices, mesh.devices)
        and sharding.is_equivalent_to(NamedSharding(mesh, spec), array.ndim)
    )


def _require_head_wing_basis_pair(
    charge_binding, transverse_binding, layout: PhotonBasisLayout,
):
    """Authenticate the two existing bases before numerical extraction."""
    from gw.wavefunction_bundle import AuthenticatedWavefunctions

    for binding, role, channel in (
            (charge_binding, "charge", 0),
            (transverse_binding, "transverse", 1)):
        if not isinstance(binding, AuthenticatedWavefunctions):
            raise TypeError(
                "static gauge head wings require host-only "
                f"AuthenticatedWavefunctions for {role}; got "
                f"{type(binding).__name__}")
        receipt = binding.receipt
        if receipt.role != role:
            raise ValueError(
                f"static gauge {role} binding carries role={receipt.role!r}")
        receipt.assert_matches_carrier(
            binding.wavefunctions,
            where=f"static gauge {role} head-wing basis")
        if (int(receipt.n_rmu_logical) != layout.logical_extent(channel)
                or int(receipt.n_rmu_padded) != layout.padded_extent(channel)):
            raise ValueError(
                f"static gauge {role} basis extent logical/padded="
                f"{receipt.n_rmu_logical}/{receipt.n_rmu_padded} does not "
                f"match photon layout {layout.logical_extent(channel)}/"
                f"{layout.padded_extent(channel)}")
    charge_binding.receipt.assert_same_wfn_manifold(
        transverse_binding.receipt,
        where="static gauge charge/transverse head-wing bases")
    if (charge_binding.wavefunctions.layout
            != transverse_binding.wavefunctions.layout):
        raise ValueError(
            "static gauge charge/transverse head-wing bases use different "
            f"carrier layouts: {charge_binding.wavefunctions.layout!r} vs "
            f"{transverse_binding.wavefunctions.layout!r}")
    return charge_binding.wavefunctions, transverse_binding.wavefunctions


def require_retained_alpha_first_order_head_wings(
    carriers: RetainedAlphaFirstOrderHeadWingCarriers,
    mesh_xy: Mesh,
) -> RetainedAlphaFirstOrderHeadWingCarriers:
    """Validate the bounded carrier's receipts, shape, dtype, and sharding."""
    if not isinstance(carriers, RetainedAlphaFirstOrderHeadWingCarriers):
        raise TypeError(
            "first-order head wings require "
            "RetainedAlphaFirstOrderHeadWingCarriers; got "
            f"{type(carriers).__name__}")
    carriers.layout.assert_mesh(mesh_xy)
    n_omega = int(carriers.n_omega)
    n_packed = int(carriers.layout.packed_extent)
    if n_omega < 1 or int(carriers.nb_logical) < 1:
        raise ValueError("first-order head-wing extents must be positive")
    arrays = (
        (carriers.Y_x, "Y_x", (n_omega, 2, 4, n_packed),
         P(None, None, None, "x")),
        (carriers.Z_y, "Z_y", (n_omega, 2, n_packed, 4),
         P(None, None, "y", None)),
    )
    for array, name, shape, spec in arrays:
        if tuple(array.shape) != shape:
            raise ValueError(f"{name} shape {array.shape} != {shape}")
        if np.dtype(array.dtype) != np.dtype(np.complex128):
            raise TypeError(f"{name} dtype {array.dtype} != complex128")
        if not _same_mesh_sharding(array, mesh_xy, spec):
            raise ValueError(f"{name} must have production sharding {spec}")

    from file_io.wfn_basis import WavefunctionBasisReceipt
    charge = carriers.charge_basis_receipt
    transverse = carriers.transverse_basis_receipt
    if (not isinstance(charge, WavefunctionBasisReceipt)
            or not isinstance(transverse, WavefunctionBasisReceipt)):
        raise TypeError("first-order head-wing carriers require basis receipts")
    if charge.role != "charge" or transverse.role != "transverse":
        raise ValueError("first-order head-wing basis roles are inconsistent")
    charge.assert_same_wfn_manifold(
        transverse, where="issued static gauge head-wing carriers")
    for receipt, channel in ((charge, 0), (transverse, 1)):
        if (int(receipt.n_rmu_logical)
                != carriers.layout.logical_extent(channel)
                or int(receipt.n_rmu_padded)
                != carriers.layout.padded_extent(channel)):
            raise ValueError(
                "issued static gauge head-wing receipt/layout extents differ")
    return carriers


def build_retained_alpha_first_order_head_wings(
    first_order,
    charge_binding,
    transverse_binding,
    *,
    layout: PhotonBasisLayout,
    mesh: Mesh,
) -> RetainedAlphaFirstOrderHeadWingCarriers:
    r"""Populate local-alpha Y/Z algebra from the width-eight retained jet.

    The function consumes existing authenticated centroid bundles.  It does
    not load a WFN, unfold symmetry, perform an FFT, rebuild a centroid basis,
    or transform an entire wavefunction bundle for a gamma vertex.  It also
    does not replace the exact finite-transfer current endpoint or claim that
    the local alpha insertion is a complete same-Hamiltonian current wing.
    """
    from gw.qsgw_head import (
        StaticGaugeFirstOrderComponent, head_wings_sharded)

    if not isinstance(layout, PhotonBasisLayout):
        raise TypeError("first-order head wings require PhotonBasisLayout")
    layout.assert_mesh(mesh)
    if not isinstance(first_order, StaticGaugeFirstOrderComponent):
        raise TypeError(
            "first-order head wings require StaticGaugeFirstOrderComponent; "
            f"got {type(first_order).__name__}")
    wfns_charge, wfns_transverse = _require_head_wing_basis_pair(
        charge_binding, transverse_binding, layout)

    p = jnp.asarray(first_order.energy_scaled_d1_raw, dtype=jnp.complex128)
    if (p.ndim != 5 or tuple(p.shape[:2]) != (2, 4)
            or p.shape[-2] != p.shape[-1]):
        raise ValueError(
            "first-order energy-scaled jet must be "
            f"(2,4,nk,nb,nb); got {p.shape}")
    nk, storage = int(p.shape[2]), int(p.shape[-1])
    logical = int(first_order.nb_logical)
    omega = jnp.atleast_1d(jnp.asarray(
        first_order.omegas_ry, dtype=jnp.complex128))
    n_omega = int(omega.shape[0])
    if not (0 < logical <= storage):
        raise ValueError(
            f"first-order head manifold logical/storage={logical}/{storage}")
    if (int(first_order.nk_tot) != nk or int(first_order.nspin) < 1
            or int(first_order.normalization_nspinor) < 1):
        raise ValueError(
            "first-order head normalization metadata is invalid: "
            f"nk={nk}/{first_order.nk_tot}, nspin={first_order.nspin}, "
            "normalization_nspinor="
            f"{first_order.normalization_nspinor}")
    if tuple(first_order.S_first_first.shape) != (n_omega, 2, 2, 4, 4):
        raise ValueError(
            "first-order S/omega axes disagree: "
            f"S={first_order.S_first_first.shape}, n_omega={n_omega}")

    for role, binding, wfns in (
            ("charge", charge_binding, wfns_charge),
            ("transverse", transverse_binding, wfns_transverse)):
        start, stop = binding.receipt.band_interval
        if start != 0 or stop < logical:
            raise ValueError(
                f"static gauge {role} basis band interval [{start},{stop}) "
                f"does not cover head manifold [0,{logical})")
        if (int(wfns.enk.shape[0]) != nk
                or int(wfns.enk.shape[1]) < storage
                or tuple(wfns.occ.shape) != tuple(wfns.enk.shape)):
            raise ValueError(
                f"static gauge {role} carrier energy/occupation shape "
                f"{wfns.enk.shape}/{wfns.occ.shape} does not cover "
                f"({nk},{storage})")
    energies = wfns_charge.enk[:, :storage]
    occupations = wfns_charge.occ[:, :storage]
    p_flat = p.reshape(8, nk, storage, storage)

    native_y = []
    native_z = []
    families = (wfns_charge, wfns_transverse,
                wfns_transverse, wfns_transverse)
    for body_channel, wfns in enumerate(families):
        y_raw, z_raw = head_wings_sharded(
            p_flat, wfns, energies, occupations, omega,
            mesh=mesh, nb_logical=logical,
            nk_tot=int(first_order.nk_tot),
            nspin=int(first_order.nspin),
            nspinor=int(first_order.normalization_nspinor),
            eta_ry=float(first_order.eta_ry),
            body_lorentz_channel=body_channel,
        )
        extent = layout.padded_extent(body_channel)
        if (tuple(y_raw.shape) != (n_omega, 8, extent)
                or tuple(z_raw.shape) != (n_omega, extent, 8)):
            raise ValueError(
                f"head-wing channel {body_channel} returned Y/Z shapes "
                f"{y_raw.shape}/{z_raw.shape}, expected "
                f"{(n_omega, 8, extent)}/{(n_omega, extent, 8)}")
        y_head_rows = y_raw.reshape(n_omega, 2, 4, extent)
        y_head_rows = y_head_rows.reshape(n_omega * 2, 4, extent)
        z_head_rows = z_raw.reshape(n_omega, extent, 2, 4)
        z_head_rows = jnp.transpose(z_head_rows, (0, 2, 3, 1))
        z_head_rows = z_head_rows.reshape(n_omega * 2, 4, extent)
        native_y.append(y_head_rows)
        native_z.append(z_head_rows)

    packed_y = pack_photon_head_body_vectors(
        tuple(native_y), layout, mesh, axis_name="x")
    packed_z = pack_photon_head_body_vectors(
        tuple(native_z), layout, mesh, axis_name="y")
    Y_x = packed_y.reshape(n_omega, 2, 4, layout.packed_extent)
    Z_y = packed_z.reshape(n_omega, 2, 4, layout.packed_extent)
    Z_y = jnp.transpose(Z_y, (0, 1, 3, 2))

    carriers = RetainedAlphaFirstOrderHeadWingCarriers(
        layout=layout, Y_x=Y_x, Z_y=Z_y,
        charge_basis_receipt=charge_binding.receipt,
        transverse_basis_receipt=transverse_binding.receipt,
        nb_logical=logical, n_omega=n_omega,
        _producer_token=_RETAINED_ALPHA_HEAD_WING_TOKEN,
    )
    return require_retained_alpha_first_order_head_wings(carriers, mesh)


def require_charge_hall_cubature_response(
    response: ChargeHallCubatureResponse, mesh_xy: Mesh,
) -> ChargeHallCubatureResponse:
    """Check the bounded model's support, dtype and sharding."""
    if not isinstance(response, ChargeHallCubatureResponse):
        raise TypeError(
            "charge_hall_cubature requires ChargeHallCubatureResponse; got "
            f"{type(response).__name__}")
    if response.capability is not StaticGaugeResponseCapability.CHARGE_HALL_CUBATURE:
        raise ValueError("charge/Hall producer cannot issue a FULL capability")
    response.availability.require_for(response.capability)
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
        raise ValueError("charge_hall_cubature sigma_H is not finite")
    if np.any(S[:, :, 0, 1:] != 0.0) or np.any(S[:, :, 1:, :] != 0.0):
        raise ValueError("charge_hall_cubature S has non-charge support")
    from gw.head_correction import static_gauge_tensor_residuals
    structural_ward, structural_hermiticity = (
        static_gauge_tensor_residuals(S))
    residuals = np.asarray((
        response.ward_residual, response.hermiticity_residual,
        response.wing_reciprocity_residual,
        structural_ward, structural_hermiticity), dtype=np.float64)
    if np.any(~np.isfinite(residuals)) or np.any(residuals < 0.0):
        raise ValueError("charge/Hall response has invalid residuals")
    if max(response.ward_residual, structural_ward) > 1.0e-8:
        raise ValueError("charge/Hall response violates the static Ward gate")
    if max(response.hermiticity_residual,
           structural_hermiticity) > 1.0e-10:
        raise ValueError("charge/Hall response violates the Hermiticity gate")
    if response.wing_reciprocity_residual > 1.0e-10:
        raise ValueError("charge/Hall response violates wing reciprocity")
    return response


def _replicated(value, mesh: Mesh, *, dtype):
    return jax.device_put(
        np.asarray(value, dtype=dtype), NamedSharding(mesh, P()))


def _channel_zeros(nq: int, extent: int, mesh: Mesh, axis: str):
    return jax.device_put(
        np.zeros((int(nq), int(extent)), dtype=np.complex128),
        NamedSharding(mesh, P(None, axis)))


def build_charge_hall_cubature_response(
    wfns,
    *,
    input_dir: str,
    mesh: Mesh,
    wfn,
    meta,
    config,
    layout: PhotonBasisLayout,
    hall_transaction,
    wfn_fingerprint_binding=None,
) -> ChargeHallCubatureResponse:
    r"""Compose charge CC and Hall CT/TC response at omega=0.

    The scalar response is evaluated once by
    :func:`gw.qsgw_head.build_dft_head_response`.  Its two in-plane velocity
    rows become the qx/qy derivatives of the charge head and charge-only
    wings.  Three exact-zero transverse vectors are passed to the canonical
    packer; no current wing is inferred.  The Hall input must be the full-BZ
    result of :func:`gw.qsgw_head.static_gauge_hall_transaction`.
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
        raise TypeError("charge/Hall response requires PhotonBasisLayout")
    layout.assert_mesh(mesh)
    if not isinstance(hall_transaction, StaticGaugeHallTransaction):
        raise TypeError(
            "charge/Hall response requires the full-BZ Hall transaction")

    wfn_fp = _canonical_wfn_sha256(
        wfn_fingerprint(wfn)
        if wfn_fingerprint_binding is None
        else fingerprint_from_binding(wfn_fingerprint_binding, wfn))
    start, stop = int(meta.b_id_0), int(meta.b_id_4_chi_user)
    if (int(hall_transaction.band_start), int(hall_transaction.band_stop)) != (
            start, stop):
        raise ValueError(
            "charge and Hall responses use different band manifolds: "
            f"charge=[{start},{stop}), Hall=[{hall_transaction.band_start},"
            f"{hall_transaction.band_stop})")
    if _canonical_wfn_sha256(hall_transaction.wfn_fingerprint) != wfn_fp:
        raise ValueError("charge and Hall responses use different WFN identities")
    direct = build_dft_head_response(
        wfns, (0.0 + 0.0j,), input_dir=input_dir, mesh=mesh,
        wfn=wfn, meta=meta, config=config,
        wfn_fingerprint_binding=wfn_fingerprint_binding)
    if direct.Y_x is None or direct.Z_y is None:
        raise ValueError("incumbent charge response returned no body wings")
    charge_extent = int(layout.padded_extent(0))
    if (tuple(direct.Y_x.shape) != (1, 3, charge_extent)
            or tuple(direct.Z_y.shape) != (1, charge_extent, 3)):
        raise ValueError(
            "charge response/layout mismatch: "
            f"Y={direct.Y_x.shape}, Z={direct.Z_y.shape}, "
            f"charge padded extent={charge_extent}")

    charge_y = direct.Y_x[0, :2, :]
    charge_z = jnp.transpose(direct.Z_y[0, :, :2], (1, 0))
    zeros_x = tuple(
        _channel_zeros(2, layout.padded_extent(A), mesh, "x")
        for A in range(1, 4))
    zeros_y = tuple(
        _channel_zeros(2, layout.padded_extent(A), mesh, "y")
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
    sigma_host = np.asarray(
        jax.device_get(hall_transaction.sigma_H), dtype=np.float64)
    if sigma_host.shape != (3,) or not np.all(np.isfinite(sigma_host)):
        raise ValueError("Hall transaction has an invalid sigma_H")
    sigma_H = _replicated(sigma_host, mesh, dtype=np.float64)
    # At static imaginary frequency the Adler--Wiser weight is real, hence
    # the two incumbent wing orientations obey Z[b,mu]=conj(Y[b,mu]).  Move
    # only this O(N_mu) vector to Y sharding for a scalar certificate.
    charge_z_x = jax.device_put(
        charge_z, NamedSharding(mesh, P(None, "x")))
    wing_delta = jnp.max(jnp.abs(charge_y - jnp.conj(charge_z_x)))
    wing_scale = jnp.maximum(
        jnp.maximum(jnp.max(jnp.abs(charge_y)),
                    jnp.max(jnp.abs(charge_z_x))), 1.0e-300)
    wing_reciprocity = float(
        np.asarray(jax.device_get(wing_delta / wing_scale)))
    ward, hermiticity = static_gauge_tensor_residuals(S_direct)

    availability = CHARGE_HALL_CUBATURE_AVAILABILITY
    response = ChargeHallCubatureResponse(
        capability=StaticGaugeResponseCapability.CHARGE_HALL_CUBATURE,
        availability=availability, layout=layout,
        S_direct=S_direct, sigma_H=sigma_H, Y_x=Y_x, Z_y=Z_y,
        ward_residual=float(ward), hermiticity_residual=float(hermiticity),
        wing_reciprocity_residual=wing_reciprocity,
        _producer_token=_PRODUCER_TOKEN,
    )
    return require_charge_hall_cubature_response(response, mesh)


def require_full_static_gauge_availability(
    availability: StaticGaugeTermAvailability,
) -> None:
    """The explicit refusal seam for consumers that claim FULL closure."""
    availability.require_for(StaticGaugeResponseCapability.FULL_STATIC_GAUGE)


__all__ = [
    "CHARGE_HALL_CUBATURE_AVAILABILITY",
    "ChargeHallCubatureResponse",
    "RetainedAlphaFirstOrderHeadWingCarriers",
    "StaticGaugeResponseCapability",
    "StaticGaugeTermAvailability",
    "StaticGaugeTermStatus",
    "build_charge_hall_cubature_response",
    "build_retained_alpha_first_order_head_wings",
    "require_charge_hall_cubature_response",
    "require_full_static_gauge_availability",
    "require_retained_alpha_first_order_head_wings",
]
