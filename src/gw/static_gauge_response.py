"""Inputs for bounded static four-current cubature responses.

``charge_hall_cubature`` is a deliberately restricted effective response:
the existing scalar-density head/wings at omega=0 plus the separately computed
Hall Chern--Simons term.  Ordinary current response,
contact and complement-space closure are omitted *by model*, never stored as
accidental zeros and never promoted to ``full_static_gauge``.

The normalized retained-bubble capability uses the same carrier and numerical
consumer, but declares its larger four-current support and its missing
contact/complement pieces explicitly.  A capability is data, never inferred
from zero entries.

This module owns no WFN load, symmetry unfold, current, FFT or body-response
implementation.  It composes the existing routines and retains only O(N_mu)
wing carriers.
"""
from __future__ import annotations

from dataclasses import dataclass, fields
import enum

import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

from common.collectives import device_put_process_local
from common.four_current_model import resolve_four_current_representation
from gw.photon_layout import PhotonBasisLayout, pack_photon_channel_vectors


class StaticGaugeResponseCapability(str, enum.Enum):
    CHARGE_HALL_CUBATURE = "charge_hall_cubature"
    ISOMETRIC_RETAINED_BUBBLE = "isometric_retained_bubble"
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
        if capability is StaticGaugeResponseCapability.CHARGE_HALL_CUBATURE:
            required = {
                "cc_q2", "ct_q1", "tc_q1", "y_charge", "z_charge"}
        elif capability is (
                StaticGaugeResponseCapability.ISOMETRIC_RETAINED_BUBBLE):
            required = {
                "cc_q2", "ct_q1", "tc_q1", "ct_q2", "tc_q2", "tt_q2",
                "y_charge", "y_current", "z_charge", "z_current",
            }
        else:
            raise ValueError(
                f"unsupported bounded static-gauge capability {capability!r}")
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


def _isometric_retained_bubble_availability() -> StaticGaugeTermAvailability:
    complete = StaticGaugeTermStatus.COMPLETE
    omitted = StaticGaugeTermStatus.OMITTED_BY_MODEL
    return StaticGaugeTermAvailability(
        cc_q2=complete, ct_q1=complete, tc_q1=complete,
        ct_q2=complete, tc_q2=complete,
        tt_q0=omitted, tt_q1=omitted, tt_q2=complete,
        y_charge=complete, y_current=complete,
        z_charge=complete, z_current=complete,
        contact_q0=omitted, contact_q2=omitted,
        complement_space=omitted,
    )


ISOMETRIC_RETAINED_BUBBLE_AVAILABILITY = (
    _isometric_retained_bubble_availability())
_PRODUCER_TOKEN = object()
_SOURCE_TOKEN = object()

ISOMETRIC_RETAINED_BUBBLE_APPROXIMATION = (
    "isometric_retained_bubble_no_contact_no_complement_v1")
ISOMETRIC_ENDPOINT_JET_CONVENTION = (
    "uniform_gauge_endpoint_jet/v1"
    "|kinetic_balance_lift=isometric"
    "|bra_K=k-q:q1=-dK,q2=+d2K"
    "|transfer_axes=cartesian"
    "|q2=analytic_isometric_plus_icl_vnl")
DISTINCT_BODY_LEG_PLACEMENT = (
    "head_wings_distinct_faces/v1|Y=bra_vertex(B,0)|Z=ket_vertex(0,B)")


def _canonical_wfn_sha256(value) -> str:
    value = str(value).strip()
    if (len(value) != 64
            or any(c not in "0123456789abcdef" for c in value)):
        raise ValueError(
            "charge/Hall WFN identifier must be 64 lowercase hex")
    return value


@dataclass(frozen=True)
class StaticGaugeCubatureResponse:
    r"""Inputs for ``R(q)=q_a H_a(sigma_H)+q_a q_b S_ab``.

    Capability declares the support of ``S_direct`` and the wings.  Charge+
    Hall has charge-only S/Y/Z; the isometric retained bubble has all four
    Lorentz fields but explicitly omits contact and complement closure.  The
    Hall tensor is intentionally not stored: :func:`static_hall_linear_response`
    constructs it from ``sigma_H`` at the numerical kernel that consumes it.
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
    charge_representation: str
    spatial_current_representation: str | None
    response_fingerprint: str | None
    approximation: str
    _producer_token: object

    def __post_init__(self) -> None:
        if self._producer_token is not _PRODUCER_TOKEN:
            raise TypeError(
                "StaticGaugeCubatureResponse is issued only by a registered "
                "bounded static-gauge response producer")


# Compatibility import for callers/tests written before the bounded carrier
# gained a second capability.  This is the same class, not a sibling record.
ChargeHallCubatureResponse = StaticGaugeCubatureResponse


@dataclass(frozen=True)
class IsometricRetainedBubbleSource:
    """Persistable response source after the expensive endpoint transaction.

    ``energy_scaled_d1_raw`` is the width-eight ``(a,I)`` head jet used by
    the canonical wing contraction.  ``S_direct`` is the retained-bubble q2
    coefficient at omega=0.  Contact and complement arrays are deliberately
    absent; their omission is carried by ``availability`` and
    ``approximation`` rather than encoded as zeros.
    """

    energy_scaled_d1_raw: jax.Array  # (2,4,nk,nb,nb), P(None,None,None,x,y)
    S_direct: jax.Array              # (2,2,4,4), replicated
    sigma_H: jax.Array               # (3,), replicated
    availability: StaticGaugeTermAvailability
    charge_representation: str
    spatial_current_representation: str
    endpoint_jet_convention: str
    hamiltonian_config_operator_fingerprint: str
    source_fingerprint: str
    wfn_fingerprint: str
    band_start: int
    band_stop: int
    nk_tot: int
    charge_ward_residual: float
    ward_residual: float
    hermiticity_residual: float
    ordered_curvature_residual: float
    q2_symmetry_residual: float
    approximation: str
    _source_token: object

    def __post_init__(self) -> None:
        if self._source_token is not _SOURCE_TOKEN:
            raise TypeError(
                "IsometricRetainedBubbleSource is issued only by the "
                "registered producer or immutable artifact loader")


def _source_fingerprint(
    *, operator_fingerprint: str, wfn_fingerprint: str,
    band_start: int, band_stop: int, nk_tot: int,
) -> str:
    """Bind the endpoint representation/jet semantics before persistence."""
    import hashlib
    from common.parallel_transport import fingerprint_update_value

    digest = hashlib.sha256()
    digest.update(b"lorrax.isometric_retained_bubble_source/v1\0")
    for label, value in (
        ("operator", operator_fingerprint),
        ("wfn", wfn_fingerprint),
        ("band_interval", f"{int(band_start)}:{int(band_stop)}"),
        ("nk_tot", str(int(nk_tot))),
        ("endpoint_jet", ISOMETRIC_ENDPOINT_JET_CONVENTION),
        ("approximation", ISOMETRIC_RETAINED_BUBBLE_APPROXIMATION),
    ):
        fingerprint_update_value(digest, label, value)
    return "sha256:" + digest.hexdigest()


def _issue_isometric_retained_bubble_source(**fields):
    """Private common constructor for the producer and SlabIO loader."""
    return IsometricRetainedBubbleSource(
        **fields, _source_token=_SOURCE_TOKEN)


def require_isometric_retained_bubble_source(
    source: IsometricRetainedBubbleSource, mesh_xy: Mesh,
) -> IsometricRetainedBubbleSource:
    """Validate the bounded source without gathering its band-square jet."""
    if not isinstance(source, IsometricRetainedBubbleSource):
        raise TypeError(
            "normalized retained-bubble response requires "
            f"IsometricRetainedBubbleSource; got {type(source).__name__}")
    source.availability.require_for(
        StaticGaugeResponseCapability.ISOMETRIC_RETAINED_BUBBLE)
    from common.four_current_model import (
        ISOMETRIC_KINETIC_BALANCE_CHARGE_REPRESENTATION,
        ISOMETRIC_KINETIC_BALANCE_SPATIAL_CURRENT_REPRESENTATION,
    )
    if (source.charge_representation !=
            ISOMETRIC_KINETIC_BALANCE_CHARGE_REPRESENTATION
            or source.spatial_current_representation !=
            ISOMETRIC_KINETIC_BALANCE_SPATIAL_CURRENT_REPRESENTATION):
        raise ValueError(
            "retained-bubble source is not in the normalized isometric "
            "four-current representation")
    if source.endpoint_jet_convention != ISOMETRIC_ENDPOINT_JET_CONVENTION:
        raise ValueError("retained-bubble endpoint-jet convention differs")
    if source.approximation != ISOMETRIC_RETAINED_BUBBLE_APPROXIMATION:
        raise ValueError("retained-bubble approximation declaration differs")
    from gw.head_correction import require_canonical_operator_fingerprint
    require_canonical_operator_fingerprint(
        source.hamiltonian_config_operator_fingerprint,
        gate="isometric_retained_bubble_operator_fingerprint")
    require_canonical_operator_fingerprint(
        source.source_fingerprint,
        gate="isometric_retained_bubble_source_fingerprint")
    _canonical_wfn_sha256(source.wfn_fingerprint)

    expected_source_fingerprint = _source_fingerprint(
        operator_fingerprint=source.hamiltonian_config_operator_fingerprint,
        wfn_fingerprint=source.wfn_fingerprint,
        band_start=int(source.band_start), band_stop=int(source.band_stop),
        nk_tot=int(source.nk_tot))
    if source.source_fingerprint != expected_source_fingerprint:
        raise ValueError(
            "retained-bubble source fingerprint does not bind its current "
            "endpoint-jet fields")

    logical = int(source.band_stop) - int(source.band_start)
    if int(source.band_start) != 0 or logical <= 0 or int(source.nk_tot) <= 0:
        raise ValueError("retained-bubble source has an invalid state manifold")
    storage = int(source.energy_scaled_d1_raw.shape[-1])
    if tuple(source.energy_scaled_d1_raw.shape) != (
            2, 4, int(source.nk_tot), storage, storage):
        raise ValueError(
            "retained-bubble P jet must be (2,4,nk,nb,nb); got "
            f"{source.energy_scaled_d1_raw.shape}")
    if storage < logical:
        raise ValueError("retained-bubble P jet does not cover logical bands")
    arrays = (
        (source.energy_scaled_d1_raw, "energy_scaled_d1_raw",
         np.complex128, P(None, None, None, "x", "y")),
        (source.S_direct, "S_direct", np.complex128, P()),
        (source.sigma_H, "sigma_H", np.float64, P()),
    )
    if tuple(source.S_direct.shape) != (2, 2, 4, 4):
        raise ValueError(f"S_direct shape {source.S_direct.shape} != (2,2,4,4)")
    if tuple(source.sigma_H.shape) != (3,):
        raise ValueError(f"sigma_H shape {source.sigma_H.shape} != (3,)")
    for array, name, dtype, spec in arrays:
        if np.dtype(array.dtype) != np.dtype(dtype):
            raise TypeError(f"{name} dtype {array.dtype} != {np.dtype(dtype)}")
        if not _same_mesh_sharding(array, mesh_xy, spec):
            raise ValueError(f"{name} must have production sharding {spec}")
    residuals = np.asarray((
        source.charge_ward_residual, source.ward_residual,
        source.hermiticity_residual, source.ordered_curvature_residual,
        source.q2_symmetry_residual), dtype=np.float64)
    if np.any(~np.isfinite(residuals)) or np.any(residuals < 0.0):
        raise ValueError("retained-bubble source has invalid residuals")
    return source


def build_isometric_retained_bubble_source(
    uniform_gauge, parallel_transport,
    *, wfn, sym, band_start: int, band_stop: int, mesh: Mesh,
) -> IsometricRetainedBubbleSource:
    """Compose the canonical isometric endpoint transaction once.

    WFN/VNL/FFT work remains in the preprocessing driver.  This function
    consumes its sealed uniform-gauge result plus the existing PT owner and
    persists only the width-eight first jet, small q2 coefficient and Hall
    vector needed later by the centroid-wing producer.
    """
    from common.bispinor_init import ISOMETRIC_KINETIC_BALANCE_LIFT
    from common.four_current_model import resolve_four_current_representation
    from common.parallel_transport import wfn_fingerprint
    from common.mtxel_sweep import UniformGaugeMatrixElements
    from gw.gw_config import BispinorGWMode
    from gw.head_correction import (
        canonicalize_static_gauge_q2_tensor,
        require_canonical_operator_fingerprint,
        static_gauge_tensor_residuals,
    )
    from gw.qsgw_head import (
        ParallelTransportHeadData,
        static_gauge_full_bz_state_tables,
        static_gauge_hall_transaction,
        static_gauge_second_order_component_sharded,
    )

    if not isinstance(uniform_gauge, UniformGaugeMatrixElements):
        raise TypeError(
            "retained-bubble source requires the complete uniform-gauge "
            "endpoint transaction")
    if (uniform_gauge.dgamma_dq_raw is None
            or uniform_gauge.d2gamma_dq2_raw is None):
        raise ValueError(
            "retained-bubble source requires canonical q1 and q2 endpoint jets")
    if not isinstance(parallel_transport, ParallelTransportHeadData):
        raise TypeError(
            "retained-bubble source requires the canonical PT link artifact")
    if parallel_transport.coefficient_frame != (
            "source_pauli_coefficient_frame_v1"):
        raise ValueError(
            "retained-bubble source requires source-Pauli coefficient-frame "
            "links; isometric endpoint q1/q2 already contains dU/dK and an "
            "isometric-link connection would double count it")
    start, stop = int(band_start), int(band_stop)
    logical = stop - start
    if start != 0 or logical <= 0:
        raise ValueError("retained-bubble band interval must be [0,stop)")
    if int(parallel_transport.nb_logical) != logical:
        raise ValueError(
            "endpoint jet and PT links use different band manifolds: "
            f"jet=[{start},{stop}), PT=[0,{parallel_transport.nb_logical})")
    representation = resolve_four_current_representation(
        True, BispinorGWMode.FULL_STATIC_COHSEX)
    if (representation.charge_lift != ISOMETRIC_KINETIC_BALANCE_LIFT
            or representation.current_lift != ISOMETRIC_KINETIC_BALANCE_LIFT):
        raise AssertionError("FULL_STATIC_COHSEX did not resolve isometric")

    energies, occupations = static_gauge_full_bz_state_tables(
        wfn=wfn, sym=sym, band_start=start, band_stop=stop)
    second = static_gauge_second_order_component_sharded(
        uniform_gauge.gamma_raw,
        uniform_gauge.dgamma_dq_raw,
        uniform_gauge.d2gamma_dq2_raw,
        parallel_transport.forward_links,
        parallel_transport.forward_neighbors,
        energies,
        occupations,
        (0.0 + 0.0j,),
        mesh=mesh,
        kgrid=tuple(int(v) for v in wfn.kgrid),
        bvec_cart=(np.asarray(wfn.bvec, dtype=np.float64)
                   * float(wfn.blat)),
        nb_logical=logical,
        cell_volume=float(wfn.cell_volume),
        nk_tot=int(sym.nk_tot),
        nspin=int(wfn.nspin),
        nspinor=int(wfn.nspinor),
        eta_ry=0.0,
    )
    hall = static_gauge_hall_transaction(
        uniform_gauge, wfn=wfn, sym=sym, band_start=start,
        band_stop=stop, mesh=mesh)
    operator_fingerprint = require_canonical_operator_fingerprint(
        uniform_gauge.hamiltonian_config_operator_fingerprint,
        gate="isometric_retained_bubble_uniform_operator")
    if hall.hamiltonian_config_operator_fingerprint != operator_fingerprint:
        raise ValueError(
            "Hall and endpoint-jet transactions have different operator "
            "fingerprints")
    wfn_fp = _canonical_wfn_sha256(wfn_fingerprint(wfn))
    if hall.wfn_fingerprint != wfn_fp:
        raise ValueError("Hall and endpoint-jet transactions use different WFN")

    S_direct = canonicalize_static_gauge_q2_tensor(
        second.S_bubble_q2_coefficient_cart[0])
    ward, hermiticity = static_gauge_tensor_residuals(S_direct)
    values = jax.device_get((
        second.first_order.charge_ward_residual,
        second.ordered_curvature_residual,
        second.q2_symmetry_residual,
    ))
    source = _issue_isometric_retained_bubble_source(
        energy_scaled_d1_raw=jnp.asarray(
            second.first_order.energy_scaled_d1_raw,
            dtype=jnp.complex128),
        S_direct=jnp.asarray(S_direct, dtype=jnp.complex128),
        sigma_H=jnp.asarray(hall.sigma_H, dtype=jnp.float64),
        availability=ISOMETRIC_RETAINED_BUBBLE_AVAILABILITY,
        charge_representation=representation.charge_representation,
        spatial_current_representation=(
            representation.spatial_current_representation),
        endpoint_jet_convention=ISOMETRIC_ENDPOINT_JET_CONVENTION,
        hamiltonian_config_operator_fingerprint=operator_fingerprint,
        source_fingerprint=_source_fingerprint(
            operator_fingerprint=operator_fingerprint,
            wfn_fingerprint=wfn_fp, band_start=start, band_stop=stop,
            nk_tot=int(sym.nk_tot)),
        wfn_fingerprint=wfn_fp,
        band_start=start, band_stop=stop, nk_tot=int(sym.nk_tot),
        charge_ward_residual=float(np.asarray(values[0])),
        ward_residual=float(ward),
        hermiticity_residual=float(hermiticity),
        ordered_curvature_residual=float(np.asarray(values[1])),
        q2_symmetry_residual=float(np.asarray(values[2])),
        approximation=ISOMETRIC_RETAINED_BUBBLE_APPROXIMATION,
    )
    return require_isometric_retained_bubble_source(source, mesh)


def _same_mesh_sharding(array, mesh: Mesh, spec: P) -> bool:
    sharding = getattr(array, "sharding", None)
    return (
        isinstance(sharding, NamedSharding)
        and tuple(sharding.mesh.axis_names) == tuple(mesh.axis_names)
        and np.array_equal(sharding.mesh.devices, mesh.devices)
        and sharding.is_equivalent_to(NamedSharding(mesh, spec), array.ndim)
    )


def require_static_gauge_cubature_response(
    response: StaticGaugeCubatureResponse, mesh_xy: Mesh,
) -> StaticGaugeCubatureResponse:
    """Check one bounded model's declared support, dtype and sharding."""
    if not isinstance(response, StaticGaugeCubatureResponse):
        raise TypeError(
            "static gauge cubature requires StaticGaugeCubatureResponse; got "
            f"{type(response).__name__}")
    if response.capability not in (
            StaticGaugeResponseCapability.CHARGE_HALL_CUBATURE,
            StaticGaugeResponseCapability.ISOMETRIC_RETAINED_BUBBLE):
        raise ValueError(
            "bounded static-gauge producer cannot issue a FULL capability")
    response.availability.require_for(response.capability)
    response.layout.assert_mesh(mesh_xy)

    if not str(response.charge_representation).strip():
        raise ValueError("bounded response has no charge representation")
    if (response.capability is
            StaticGaugeResponseCapability.ISOMETRIC_RETAINED_BUBBLE):
        from common.four_current_model import (
            ISOMETRIC_KINETIC_BALANCE_CHARGE_REPRESENTATION,
            ISOMETRIC_KINETIC_BALANCE_SPATIAL_CURRENT_REPRESENTATION,
        )
        if (response.charge_representation !=
                ISOMETRIC_KINETIC_BALANCE_CHARGE_REPRESENTATION
                or response.spatial_current_representation !=
                ISOMETRIC_KINETIC_BALANCE_SPATIAL_CURRENT_REPRESENTATION):
            raise ValueError(
                "isometric retained-bubble response carries a different "
                "charge/current representation")
        from gw.head_correction import require_canonical_operator_fingerprint
        require_canonical_operator_fingerprint(
            response.response_fingerprint,
            gate="isometric_retained_bubble_fingerprint")
    elif response.response_fingerprint is not None:
        from gw.head_correction import require_canonical_operator_fingerprint
        require_canonical_operator_fingerprint(
            response.response_fingerprint,
            gate="charge_hall_bounded_response_fingerprint")
    if not str(response.approximation).strip():
        raise ValueError("bounded response has no approximation declaration")

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
    if (response.capability is
            StaticGaugeResponseCapability.CHARGE_HALL_CUBATURE
            and (np.any(S[:, :, 0, 1:] != 0.0)
                 or np.any(S[:, :, 1:, :] != 0.0))):
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


def require_charge_hall_cubature_response(
    response: StaticGaugeCubatureResponse, mesh_xy: Mesh,
) -> StaticGaugeCubatureResponse:
    """Compatibility validator narrowed to the charge-plus-Hall capability."""
    response = require_static_gauge_cubature_response(response, mesh_xy)
    if response.capability is not StaticGaugeResponseCapability.CHARGE_HALL_CUBATURE:
        raise ValueError(
            "charge_hall_cubature requires the charge_hall_cubature "
            f"capability; got {response.capability.value}")
    return response


def _bounded_response_fingerprint(
    source: IsometricRetainedBubbleSource,
    charge_receipt,
    transverse_receipt,
    layout: PhotonBasisLayout,
) -> str:
    """Bind endpoint jet, both centroid bases and distinct body placement."""
    import hashlib
    from dataclasses import fields
    from common.parallel_transport import fingerprint_update_value

    digest = hashlib.sha256()
    digest.update(b"lorrax.isometric_retained_bubble_response/v1\0")
    fingerprint_update_value(
        digest, "source_fingerprint", source.source_fingerprint)
    fingerprint_update_value(
        digest, "body_leg_placement", DISTINCT_BODY_LEG_PLACEMENT)
    fingerprint_update_value(
        digest, "layout_logical", layout.logical_extents)
    fingerprint_update_value(
        digest, "layout_padded", layout.padded_extents)
    for role, receipt in (
            ("charge", charge_receipt),
            ("transverse", transverse_receipt)):
        for item in fields(receipt):
            fingerprint_update_value(
                digest, f"{role}.{item.name}", getattr(receipt, item.name))
    return "sha256:" + digest.hexdigest()


def build_isometric_retained_bubble_response(
    source: IsometricRetainedBubbleSource,
    wfns_charge,
    wfns_transverse,
    *,
    charge_binding,
    transverse_binding,
    layout: PhotonBasisLayout,
    mesh: Mesh,
    wfn,
    meta,
) -> StaticGaugeCubatureResponse:
    """Attach canonical distinct-face body wings to the persisted head jet."""
    source = require_isometric_retained_bubble_source(source, mesh)
    if not isinstance(layout, PhotonBasisLayout):
        raise TypeError("retained-bubble response requires PhotonBasisLayout")
    layout.assert_mesh(mesh)
    from common.bispinor_init import (
        ISOMETRIC_KINETIC_BALANCE_LIFT_PROVENANCE)
    from gw.wavefunction_bundle import (
        AuthenticatedWavefunctions, with_lorentz_vertices)
    for binding, carrier, role in (
            (charge_binding, wfns_charge, "charge"),
            (transverse_binding, wfns_transverse, "transverse")):
        if not isinstance(binding, AuthenticatedWavefunctions):
            raise TypeError(
                f"retained-bubble {role} carrier lacks its authenticated "
                "WavefunctionBasisReceipt")
        if binding.wavefunctions is not carrier or binding.receipt.role != role:
            raise ValueError(
                f"retained-bubble {role} binding does not name its carrier")
        receipt = binding.receipt
        if receipt.wfn_fingerprint != source.wfn_fingerprint:
            raise ValueError(
                f"retained-bubble {role} basis and endpoint jet use "
                "different WFN identities")
        if receipt.band_interval != (
                int(source.band_start), int(source.band_stop)):
            raise ValueError(
                f"retained-bubble {role} basis band interval "
                f"{receipt.band_interval} != source "
                f"[{source.band_start},{source.band_stop})")
        if (receipt.bispinor_lift_provenance !=
                ISOMETRIC_KINETIC_BALANCE_LIFT_PROVENANCE):
            raise ValueError(
                f"retained-bubble {role} basis is not the isometric carrier")
        if carrier.layout != "face":
            raise ValueError(
                f"retained-bubble {role} wings require low-memory face "
                f"layout; got {carrier.layout!r}")

    charge_receipt = charge_binding.receipt
    transverse_receipt = transverse_binding.receipt
    expected_logical = (
        int(charge_receipt.n_rmu_logical),
        int(transverse_receipt.n_rmu_logical),
        int(transverse_receipt.n_rmu_logical),
        int(transverse_receipt.n_rmu_logical),
    )
    expected_padded = (
        int(charge_receipt.n_rmu_padded),
        int(transverse_receipt.n_rmu_padded),
        int(transverse_receipt.n_rmu_padded),
        int(transverse_receipt.n_rmu_padded),
    )
    if (layout.logical_extents != expected_logical
            or layout.padded_extents != expected_padded):
        raise ValueError(
            "PhotonBasisLayout differs from authenticated charge/transverse "
            f"centroid bases: layout={layout.logical_extents}/"
            f"{layout.padded_extents}, receipts={expected_logical}/"
            f"{expected_padded}")

    p = source.energy_scaled_d1_raw
    storage = int(p.shape[-1])
    logical = int(source.band_stop) - int(source.band_start)
    if (int(wfns_charge.enk.shape[0]) != int(source.nk_tot)
            or int(wfns_charge.enk.shape[1]) < storage
            or wfns_charge.enk.shape != wfns_transverse.enk.shape
            or wfns_charge.occ.shape != wfns_transverse.occ.shape):
        raise ValueError(
            "retained-bubble source and live charge/transverse state tables "
            "have different k/band carriers")
    energy = wfns_charge.enk[:, :storage]
    occupations = wfns_charge.occ[:, :storage]
    if (not np.array_equal(np.asarray(wfns_charge.enk),
                           np.asarray(wfns_transverse.enk))
            or not np.array_equal(np.asarray(wfns_charge.occ),
                                  np.asarray(wfns_transverse.occ))):
        raise ValueError(
            "charge/transverse centroid bundles carry different electronic "
            "state tables")

    from gw.qsgw_head import head_wings_sharded
    p_flat = p.reshape(8, int(source.nk_tot), storage, storage)
    Y_by_channel = []
    Z_by_channel = []
    for B in range(4):
        bundle = wfns_charge if B == 0 else wfns_transverse
        bra = with_lorentz_vertices(bundle, B, 0)
        ket = with_lorentz_vertices(bundle, 0, B)
        Y_B, Z_B = head_wings_sharded(
            p_flat,
            bundle,
            energy,
            occupations,
            (0.0 + 0.0j,),
            mesh=mesh,
            nb_logical=logical,
            nk_tot=int(source.nk_tot),
            nspin=int(wfn.nspin),
            nspinor=int(meta.nspinor_wfnfile),
            eta_ry=0.0,
            body_bra_wfns=bra,
            body_ket_wfns=ket,
        )
        extent = int(layout.padded_extent(B))
        Y_by_channel.append(Y_B[0].reshape(2, 4, extent))
        Z_by_channel.append(Z_B[0].reshape(extent, 2, 4))

    y_rows = []
    z_rows = []
    for a in range(2):
        y_fields = []
        z_fields = []
        for I in range(4):
            packed = pack_photon_channel_vectors(
                tuple(Y_by_channel[B][a, I][None, :]
                      for B in range(4)),
                layout, mesh, axis_name="x")
            y_fields.append(jnp.sum(packed[0], axis=0))
            packed_z = pack_photon_channel_vectors(
                tuple(Z_by_channel[B][:, a, I][None, :]
                      for B in range(4)),
                layout, mesh, axis_name="y")
            z_fields.append(jnp.sum(packed_z[0], axis=0))
        y_rows.append(jnp.stack(y_fields, axis=0))
        z_rows.append(jnp.stack(z_fields, axis=-1))
    Y_x = jnp.stack(y_rows, axis=0)
    Z_y = jnp.stack(z_rows, axis=0)

    z_as_y = device_put_process_local(
        jnp.transpose(Z_y, (0, 2, 1)),
        NamedSharding(mesh, P(None, None, "x")))
    delta = jnp.max(jnp.abs(Y_x - jnp.conj(z_as_y)))
    scale = jnp.maximum(
        jnp.maximum(jnp.max(jnp.abs(Y_x)), jnp.max(jnp.abs(z_as_y))),
        1.0e-300)
    wing_reciprocity = float(np.asarray(jax.device_get(delta / scale)))
    response = StaticGaugeCubatureResponse(
        capability=StaticGaugeResponseCapability.ISOMETRIC_RETAINED_BUBBLE,
        availability=ISOMETRIC_RETAINED_BUBBLE_AVAILABILITY,
        layout=layout,
        S_direct=source.S_direct,
        sigma_H=source.sigma_H,
        Y_x=Y_x,
        Z_y=Z_y,
        ward_residual=float(source.ward_residual),
        hermiticity_residual=float(source.hermiticity_residual),
        wing_reciprocity_residual=wing_reciprocity,
        charge_representation=source.charge_representation,
        spatial_current_representation=source.spatial_current_representation,
        response_fingerprint=_bounded_response_fingerprint(
            source, charge_receipt, transverse_receipt, layout),
        approximation=source.approximation,
        _producer_token=_PRODUCER_TOKEN,
    )
    return require_static_gauge_cubature_response(response, mesh)


def _replicated(value, mesh: Mesh, *, dtype):
    return device_put_process_local(
        np.asarray(value, dtype=dtype), NamedSharding(mesh, P()))


def _channel_zeros(nq: int, extent: int, mesh: Mesh, axis: str):
    return device_put_process_local(
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
) -> StaticGaugeCubatureResponse:
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
    charge_z_x = device_put_process_local(
        charge_z, NamedSharding(mesh, P(None, "x")))
    wing_delta = jnp.max(jnp.abs(charge_y - jnp.conj(charge_z_x)))
    wing_scale = jnp.maximum(
        jnp.maximum(jnp.max(jnp.abs(charge_y)),
                    jnp.max(jnp.abs(charge_z_x))), 1.0e-300)
    wing_reciprocity = float(
        np.asarray(jax.device_get(wing_delta / wing_scale)))
    ward, hermiticity = static_gauge_tensor_residuals(S_direct)

    availability = CHARGE_HALL_CUBATURE_AVAILABILITY
    representation = resolve_four_current_representation(
        bool(getattr(config, "bispinor", True)),
        getattr(config, "bispinor_gw", "charge_hall_cubature"))
    response = StaticGaugeCubatureResponse(
        capability=StaticGaugeResponseCapability.CHARGE_HALL_CUBATURE,
        availability=availability, layout=layout,
        S_direct=S_direct, sigma_H=sigma_H, Y_x=Y_x, Z_y=Z_y,
        ward_residual=float(ward), hermiticity_residual=float(hermiticity),
        wing_reciprocity_residual=wing_reciprocity,
        charge_representation=representation.charge_representation,
        spatial_current_representation=(
            representation.spatial_current_representation),
        response_fingerprint=None,
        approximation="charge_hall_cubature_v1",
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
    "DISTINCT_BODY_LEG_PLACEMENT",
    "ISOMETRIC_ENDPOINT_JET_CONVENTION",
    "ISOMETRIC_RETAINED_BUBBLE_APPROXIMATION",
    "ISOMETRIC_RETAINED_BUBBLE_AVAILABILITY",
    "IsometricRetainedBubbleSource",
    "StaticGaugeCubatureResponse",
    "StaticGaugeResponseCapability",
    "StaticGaugeTermAvailability",
    "StaticGaugeTermStatus",
    "build_charge_hall_cubature_response",
    "build_isometric_retained_bubble_source",
    "build_isometric_retained_bubble_response",
    "require_charge_hall_cubature_response",
    "require_static_gauge_cubature_response",
    "require_full_static_gauge_availability",
    "require_isometric_retained_bubble_source",
]
