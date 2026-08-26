"""Truthful upstream producers for static long-wave photon response jets.

``charge_hall_cubature`` is a deliberately restricted effective response:
the incumbent authenticated scalar-density head/wings at omega=0 plus the
separately authenticated Hall Chern--Simons jet.  Ordinary current response,
contact and complement-space closure are omitted *by model*, never stored as
accidental zeros and never promoted to ``full_static_gauge``.

This module owns no WFN load, symmetry unfold, current, FFT, body-response or
packing implementation.  It composes their public transactions and retains
only O(N_mu) wing carriers.
"""
from __future__ import annotations

from dataclasses import dataclass, fields
import enum
import hashlib
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

from gw.photon_layout import (
    PHOTON_BARE_PROPAGATOR,
    PHOTON_BASIS_ORDERING,
    PhotonBasisLayout,
    pack_photon_channel_vectors,
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
CHARGE_HALL_Q_CART_UNITS = "bohr^-1"
CHARGE_HALL_Q_POWERS = (0, 1, 2)
CHARGE_HALL_SIGN_CONVENTION = (
    "+i*epsilon[b,a,i]*sigma_H[b]*q[a]")
CHARGE_HALL_NORMALIZATION_CONVENTION = (
    "S_CC_includes_1/Omega;Y,Z_unscaled;"
    "head_Schur_adds_Y*D_body*Z/Omega")
CHARGE_HALL_CUBATURE_CONVENTION_ID = (
    "lorrax.static_gauge_response/charge_hall_cubature_v1"
    "|omega=0_static|energy=Rydberg|length=bohr"
    f"|lorentz=(C,Jx,Jy,Jz)|q_cart={CHARGE_HALL_Q_CART_UNITS}"
    "|Pi0=0|Pi1_CT=+i*epsilon[b,a,i]*sigma_H[b]"
    "|Pi1_TC=Pi1_CT^dagger|S_q2=charge_CC_only"
    "|ordinary_current_contact_complement=omitted_by_model"
    f"|normalization={CHARGE_HALL_NORMALIZATION_CONVENTION}"
    f"|packing={PHOTON_BASIS_ORDERING}|bare={PHOTON_BARE_PROPAGATOR}"
)
CHARGE_HALL_WING_DRESSING = (
    "direct_irreducible_unfolded;single=D_head*Y*D_body;"
    "double=Y*D_body*Z_in_head_Schur")
_TRANSACTION_FINGERPRINT_SCHEME = (
    "lorrax.static_gauge_response.transaction/v1")
_CHARGE_FINGERPRINT_SCHEME = "lorrax.static_charge_head_response/v1"
_HALL_FINGERPRINT_SCHEME = "lorrax.static_hall_response/v1"
_PRODUCER_TOKEN = object()


def _canonical_sha256(value, *, gate: str) -> str:
    # Import lazily: downstream head completion may import the enum/carrier
    # definitions from this module, so a module-level reverse import would be
    # a circular dependency.
    from gw.head_correction import require_canonical_operator_fingerprint
    return require_canonical_operator_fingerprint(value, gate=gate)


def _canonical_wfn_sha256(value) -> str:
    value = str(value).strip()
    if (len(value) != 64
            or any(c not in "0123456789abcdef" for c in value)):
        raise ValueError(
            "canonical charge/Hall WFN fingerprint must be 64 lowercase hex")
    return value


def _digest(payload: dict) -> str:
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"),
        ensure_ascii=True).encode("ascii")
    return "sha256:" + hashlib.sha256(encoded).hexdigest()


def _layout_record(layout: PhotonBasisLayout) -> dict:
    return {
        "logical_extents": [int(n) for n in layout.logical_extents],
        "padded_extents": [int(n) for n in layout.padded_extents],
        "mesh_side": int(layout.mesh_side),
        "ordering": str(layout.ordering),
        "bare_propagator": str(layout.bare_propagator),
    }


@dataclass(frozen=True)
class ChargeHallCubatureResponse:
    """Sealed direct response usable only under the named truncated model."""

    capability: StaticGaugeResponseCapability
    availability: StaticGaugeTermAvailability
    layout: PhotonBasisLayout
    Pi0_direct: jax.Array             # (4,4), replicated; exactly zero
    Pi1_direct: jax.Array             # (2,4,4), replicated Hall CT/TC
    S_direct: jax.Array               # (2,2,4,4), replicated charge CC
    sigma_H: jax.Array                # (3,), replicated real bohr^-1
    Y_x: jax.Array                    # (2,4,Npacked), P(None,None,'x')
    Z_y: jax.Array                    # (2,Npacked,4), P(None,'y',None)
    charge_response_fingerprint: str
    hall_response_fingerprint: str
    response_transaction_fingerprint: str
    wfn_fingerprint: str
    band_start: int
    band_stop: int
    convention_id: str
    wing_dressing: str
    q_cart_units: str
    q_powers: tuple[int, int, int]
    hall_sign_convention: str
    normalization_convention: str
    ward_residual: float
    hermiticity_residual: float
    wing_reciprocity_residual: float
    _producer_token: object

    def __post_init__(self) -> None:
        if self._producer_token is not _PRODUCER_TOKEN:
            raise TypeError(
                "ChargeHallCubatureResponse is issued only by "
                "build_charge_hall_cubature_response")


def _same_mesh_sharding(array, mesh: Mesh, spec: P) -> bool:
    sharding = getattr(array, "sharding", None)
    return (
        isinstance(sharding, NamedSharding)
        and tuple(sharding.mesh.axis_names) == tuple(mesh.axis_names)
        and np.array_equal(sharding.mesh.devices, mesh.devices)
        and sharding.is_equivalent_to(NamedSharding(mesh, spec), array.ndim)
    )


def require_charge_hall_cubature_response(
    response: ChargeHallCubatureResponse, mesh_xy: Mesh,
) -> ChargeHallCubatureResponse:
    """Fail closed on capability, provenance, support, dtype and sharding."""
    if not isinstance(response, ChargeHallCubatureResponse):
        raise TypeError(
            "charge_hall_cubature requires ChargeHallCubatureResponse; got "
            f"{type(response).__name__}")
    if response.capability is not StaticGaugeResponseCapability.CHARGE_HALL_CUBATURE:
        raise ValueError("charge/Hall producer cannot issue a FULL capability")
    response.availability.require_for(response.capability)
    response.layout.assert_mesh(mesh_xy)
    for value, gate in (
        (response.charge_response_fingerprint, "charge_response_fingerprint"),
        (response.hall_response_fingerprint, "hall_response_fingerprint"),
        (response.response_transaction_fingerprint,
         "response_transaction_fingerprint"),
    ):
        _canonical_sha256(value, gate=gate)
    _canonical_wfn_sha256(response.wfn_fingerprint)
    if not (0 <= int(response.band_start) < int(response.band_stop)):
        raise ValueError("charge/Hall response has an invalid band interval")
    if (response.convention_id != CHARGE_HALL_CUBATURE_CONVENTION_ID
            or response.wing_dressing != CHARGE_HALL_WING_DRESSING
            or response.q_cart_units != CHARGE_HALL_Q_CART_UNITS
            or tuple(response.q_powers) != CHARGE_HALL_Q_POWERS
            or response.hall_sign_convention != CHARGE_HALL_SIGN_CONVENTION
            or response.normalization_convention
            != CHARGE_HALL_NORMALIZATION_CONVENTION):
        raise ValueError("charge/Hall response convention record drifted")

    n_packed = int(response.layout.packed_extent)
    arrays = (
        (response.Pi0_direct, "Pi0_direct", (4, 4), np.complex128, P()),
        (response.Pi1_direct, "Pi1_direct", (2, 4, 4), np.complex128, P()),
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

    pi0 = np.asarray(jax.device_get(response.Pi0_direct))
    pi1 = np.asarray(jax.device_get(response.Pi1_direct))
    S = np.asarray(jax.device_get(response.S_direct))
    sigma = np.asarray(jax.device_get(response.sigma_H))
    if not np.array_equal(pi0, np.zeros_like(pi0)):
        raise ValueError("charge_hall_cubature Pi0 must be structurally zero")
    if np.any(S[:, :, 0, 1:] != 0.0) or np.any(S[:, :, 1:, :] != 0.0):
        raise ValueError("charge_hall_cubature S has non-charge support")
    from gw.head_correction import (
        static_gauge_tensor_residuals, static_hall_linear_response)
    expected_pi1 = np.asarray(static_hall_linear_response(sigma))
    if not np.array_equal(pi1, expected_pi1):
        raise ValueError("Pi1 is not the canonical Hall CT/TC tensor")
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
    r"""Compose authenticated charge CC and Hall CT/TC response at omega=0.

    The scalar response is evaluated once by
    :func:`gw.qsgw_head.build_dft_head_response`.  Its two in-plane velocity
    rows become the qx/qy derivatives of the charge head and charge-only
    wings.  Three exact-zero transverse vectors are passed to the canonical
    packer; no current wing is inferred.  The Hall transaction must be the
    sealed full-BZ result of :func:`gw.qsgw_head.static_gauge_hall_transaction`.
    """
    from common.parallel_transport import (
        fingerprint_from_binding, wfn_fingerprint)
    from gw.head_correction import (
        static_gauge_tensor_residuals, static_hall_linear_response)
    from gw.qsgw_head import (
        StaticGaugeHallTransaction, build_dft_head_response)
    from psp.get_dipole_mtxels import (
        authenticated_dipole_operator_fingerprint,
        resolve_vnl_velocity_sign,
    )

    if not isinstance(layout, PhotonBasisLayout):
        raise TypeError("charge/Hall response requires PhotonBasisLayout")
    layout.assert_mesh(mesh)
    if not isinstance(hall_transaction, StaticGaugeHallTransaction):
        raise TypeError(
            "charge/Hall response requires the sealed full-BZ Hall transaction")

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
    hall_operator_fp = _canonical_sha256(
        hall_transaction.hamiltonian_config_operator_fingerprint,
        gate="hall_operator_fingerprint")

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
    S_host[:, :, 0, 0] = np.asarray(
        jax.device_get(direct.S_direct[0, :2, :2]), dtype=np.complex128)
    S_direct = _replicated(S_host, mesh, dtype=np.complex128)
    sigma_host = np.asarray(
        jax.device_get(hall_transaction.sigma_H), dtype=np.float64)
    if sigma_host.shape != (3,) or not np.all(np.isfinite(sigma_host)):
        raise ValueError("Hall transaction has an invalid sigma_H")
    sigma_H = _replicated(sigma_host, mesh, dtype=np.float64)
    Pi0_direct = _replicated(np.zeros((4, 4)), mesh, dtype=np.complex128)
    Pi1_direct = _replicated(
        np.asarray(static_hall_linear_response(sigma_host)), mesh,
        dtype=np.complex128)

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

    dipole_path = str(Path(input_dir) / "dipole.h5")
    vnl_sign = resolve_vnl_velocity_sign(None, config.vnl_velocity_sign)
    dipole_fp = authenticated_dipole_operator_fingerprint(
        dipole_path, wfn=wfn, nval=int(config.nval),
        ncond=int(config.ncond), nband=int(config.nband),
        bispinor=bool(int(meta.nspinor) == 4), skip_vnl=False,
        vnl_mode="analytic", vnl_velocity_sign=vnl_sign,
        wfn_fingerprint_binding=wfn_fingerprint_binding)
    charge_fp = _digest({
        "scheme": _CHARGE_FINGERPRINT_SCHEME,
        "dipole_operator_fingerprint": dipole_fp,
        "wfn_fingerprint": wfn_fp,
        "bands": [start, stop],
        "omega_ry": "0x0.0p+0",
        "eta_ry": float(config.head.wcoul0_eta).hex(),
        "normalization_nspin": int(wfn.nspin),
        "normalization_nspinor_wfn": int(meta.nspinor_wfnfile),
        "cell_volume_bohr3": float(meta.cell_volume).hex(),
        "layout": _layout_record(layout),
    })
    hall_fp = _digest({
        "scheme": _HALL_FINGERPRINT_SCHEME,
        "producer_id": str(hall_transaction.producer_id),
        "operator_fingerprint": hall_operator_fp,
        "wfn_fingerprint": wfn_fp,
        "bands": [start, stop],
        "nk_tot": int(hall_transaction.nk_tot),
        "sigma_H_raw_bohr^-1": [float(value).hex() for value in sigma_host],
    })
    availability = CHARGE_HALL_CUBATURE_AVAILABILITY
    transaction_payload = {
        "scheme": _TRANSACTION_FINGERPRINT_SCHEME,
        "capability": StaticGaugeResponseCapability.CHARGE_HALL_CUBATURE.value,
        "convention_id": CHARGE_HALL_CUBATURE_CONVENTION_ID,
        "charge_response_fingerprint": charge_fp,
        "hall_response_fingerprint": hall_fp,
        "wfn_fingerprint": wfn_fp,
        "bands": [start, stop],
        "layout": _layout_record(layout),
        "q_cart_units": CHARGE_HALL_Q_CART_UNITS,
        "q_powers": {"Pi0_direct": 0, "Pi1_direct": 1, "S_direct": 2,
                     "Y_x": 1, "Z_y": 1},
        "hall_sign": CHARGE_HALL_SIGN_CONVENTION,
        "normalization": CHARGE_HALL_NORMALIZATION_CONVENTION,
        "availability": availability.as_tokens(),
        "wing_dressing": CHARGE_HALL_WING_DRESSING,
    }
    response = ChargeHallCubatureResponse(
        capability=StaticGaugeResponseCapability.CHARGE_HALL_CUBATURE,
        availability=availability, layout=layout,
        Pi0_direct=Pi0_direct, Pi1_direct=Pi1_direct,
        S_direct=S_direct, sigma_H=sigma_H, Y_x=Y_x, Z_y=Z_y,
        charge_response_fingerprint=charge_fp,
        hall_response_fingerprint=hall_fp,
        response_transaction_fingerprint=_digest(transaction_payload),
        wfn_fingerprint=wfn_fp,
        band_start=start, band_stop=stop,
        convention_id=CHARGE_HALL_CUBATURE_CONVENTION_ID,
        wing_dressing=CHARGE_HALL_WING_DRESSING,
        q_cart_units=CHARGE_HALL_Q_CART_UNITS,
        q_powers=CHARGE_HALL_Q_POWERS,
        hall_sign_convention=CHARGE_HALL_SIGN_CONVENTION,
        normalization_convention=CHARGE_HALL_NORMALIZATION_CONVENTION,
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
    "CHARGE_HALL_CUBATURE_CONVENTION_ID",
    "CHARGE_HALL_NORMALIZATION_CONVENTION",
    "CHARGE_HALL_Q_CART_UNITS",
    "CHARGE_HALL_Q_POWERS",
    "CHARGE_HALL_SIGN_CONVENTION",
    "CHARGE_HALL_WING_DRESSING",
    "ChargeHallCubatureResponse",
    "StaticGaugeResponseCapability",
    "StaticGaugeTermAvailability",
    "StaticGaugeTermStatus",
    "build_charge_hall_cubature_response",
    "require_charge_hall_cubature_response",
    "require_full_static_gauge_availability",
]
