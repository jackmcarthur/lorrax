"""Immutable SlabIO format for a gauge-complete static photon head.

This is the sole format owner for ``StaticGaugeHeadResponse``.  The bounded
direct tensor and Hall vector are replicated; the two body wings retain their
incumbent packed-body shardings.  SlabIO is the only transport, so neither
wing is gathered by this module.

The loader does not make ``FULL_SCREENED`` reachable.  Its sealed subtype
only proves that the response passed this exact schema and the caller's WFN,
band, body, and gauged-operator identities.  The driver remains refused until
the real VNL/contact/Hall producer is connected and reviewed.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import os
from pathlib import Path

import jax
import numpy as np
from jax.sharding import Mesh, PartitionSpec as P

from common.collectives import barrier
from common.parallel_transport import (
    WFN_FINGERPRINT_SCHEME,
    fingerprint_from_binding,
    wfn_fingerprint,
)
from file_io.slab_io import SlabIO
from gw.head_correction import (
    StaticGaugeHeadResponse,
    require_static_gauge_head_response,
)
from gw.photon_layout import (
    PHOTON_BARE_PROPAGATOR,
    PHOTON_BASIS_ORDERING,
    PhotonBasisLayout,
)


STATIC_GAUGE_HEAD_SCHEMA_VERSION = 3
STATIC_GAUGE_HEAD_CONVENTION_ID = (
    "lorrax.static_gauge_head/v3"
    "|omega=0_static"
    "|energy=Rydberg"
    "|length=bohr"
    "|fourier=zeta_G(q):FFT_backward[exp(-i*q_cart.r)*zeta(r)]"
    "|Gamma_raw=(alpha_FS/2)*dH_Pauli_Ry/dk"
    "|physical_current=j:c*Gamma_raw=(1/2)*dH_Pauli_Ry/dk"
    "|Lambda_raw=(alpha_FS/2)*d2H_Pauli_Ry/dkdk"
    "|lorentz=(C,Jx,Jy,Jz)"
    "|q_cart=bohr^-1"
    "|head=Pi_reg(q):q_a*q_b*S_direct[a,b]"
    "|hall_CT=-i*epsilon[b,a,i]*sigma_H[b]*q[a]"
    "|hall_TC=CT^dagger"
    "|sigma_H_raw=-(alpha_FS*C/(2*Omega_cell))*Im(cB);"
    "C=2/(nspin*nspinor_wfn)"
    "|normalization=S_direct_includes_1/Omega;Y,Z_unscaled;"
    "Schur_adds_YWZ/Omega"
    f"|packing={PHOTON_BASIS_ORDERING}"
    f"|bare={PHOTON_BARE_PROPAGATOR}"
    "|ibz_unfold=symmetry_maps_service"
    "|dtype=S,Y,Z:complex128;sigma_H:float64"
)

STATIC_GAUGE_HALL_SCHEMA_VERSION = 2
ISOMETRIC_RETAINED_BUBBLE_SOURCE_SCHEMA_VERSION = 4
ISOMETRIC_RETAINED_BUBBLE_SOURCE_CONVENTION_ID = (
    "lorrax.isometric_retained_bubble_source/v4"
    "|payload=energy_scaled_d1_raw+S_direct_cart+mixed_transition_tensor_cart"
    "|sharding=P(None,None,None,x,y)+replicated"
    "|mixed_CT=occupied_bra_real_2x3;CT=-i*T;TC=CT^dagger"
    "|contact=omitted_by_model|complement=omitted_by_model")

_S_DATASET = "S_direct_cart"
_Y_DATASET = "Y_cart_x"
_Z_DATASET = "Z_cart_y"
_P_SOURCE_DATASET = "energy_scaled_d1_raw"
_S_SOURCE_DATASET = "S_direct_cart"
_LOADER_TOKEN = object()


def _require_prefixed_sha256(value: str, *, field_name: str) -> str:
    value = str(value).strip()
    if (not value.startswith("sha256:")
            or len(value) != len("sha256:") + 64
            or any(c not in "0123456789abcdef" for c in value[7:])):
        raise ValueError(
            f"StaticGaugeHead {field_name} must be "
            "sha256:<64 lowercase hex>")
    return value


def _require_wfn_sha256(value: str) -> str:
    value = str(value).strip()
    if (len(value) != 64
            or any(c not in "0123456789abcdef" for c in value)):
        raise ValueError(
            "canonical StaticGaugeHead WFN fingerprint must be 64 "
            "lowercase hexadecimal characters")
    return value


def _text_i32(value: str, *, encoding: str = "utf-8") -> np.ndarray:
    return np.frombuffer(str(value).encode(encoding), dtype=np.uint8).astype(
        np.int32)


def _read_required_small(io: SlabIO, name: str, *, dtype=None):
    try:
        return io.read_small(name, dtype=dtype)
    except (KeyError, RuntimeError, ValueError) as exc:
        raise ValueError(
            "GATE static_gauge_head_schema: artifact is missing or cannot "
            f"read required field {name!r}") from exc


def _decode_i32_text(value, *, field_name: str, encoding: str) -> str:
    raw = np.asarray(value, dtype=np.int32).reshape(-1)
    if np.any(raw < 0) or np.any(raw > 255):
        raise ValueError(
            "GATE static_gauge_head_schema: "
            f"{field_name!r} contains a non-byte value")
    try:
        return raw.astype(np.uint8).tobytes().decode(encoding)
    except UnicodeDecodeError as exc:
        raise ValueError(
            "GATE static_gauge_head_schema: "
            f"{field_name!r} is not valid {encoding}") from exc


def _immutable_partial_paths(
    path: str | Path, *, artifact_name: str,
) -> tuple[Path, Path]:
    """Validate one create-once destination shared by both gauge schemas."""
    final_path = Path(path)
    partial_path = Path(str(final_path) + ".partial")
    if final_path.name.endswith(".partial"):
        raise ValueError(f"{artifact_name} final path may not end in '.partial'")
    if not final_path.parent.is_dir():
        raise FileNotFoundError(
            f"{artifact_name} parent directory does not exist: "
            f"{final_path.parent}")
    if os.path.lexists(final_path):
        raise FileExistsError(
            f"immutable {artifact_name} artifact already exists: {final_path}")
    if os.path.lexists(partial_path):
        raise FileExistsError(
            f"stale/in-flight {artifact_name} partial exists: {partial_path}")
    return final_path, partial_path


def _publish_completed_partial(
    partial_path: Path,
    final_path: Path,
    *,
    artifact_name: str,
    barrier_name: str,
) -> None:
    """Collectively hard-link-publish one completed SlabIO inode."""
    try:
        os.link(partial_path, final_path)
    except FileExistsError:
        if not os.path.samefile(partial_path, final_path):
            raise FileExistsError(
                f"immutable {artifact_name} publish collided with a "
                f"different artifact at {final_path}") from None
    barrier(barrier_name)
    try:
        os.unlink(partial_path)
    except FileNotFoundError:
        pass


@dataclass(frozen=True)
class LoadedStaticGaugeHeadResponse(StaticGaugeHeadResponse):
    """Validated response subtype issued only by the artifact loader.

    The private token distinguishes it from the public, caller-constructible
    response record.  It is not authorization to enable production FULL; the
    independent driver refusal remains unconditional.
    """

    convention_id: str
    artifact_path: str
    wfn_fingerprint_scheme: str
    wfn_fingerprint: str
    band_start: int
    band_stop: int
    body_response_fingerprint: str
    source_write_ibz_only: bool
    source_low_mem_bands: bool
    _loader_token: object = field(repr=False, compare=False)

    def __post_init__(self) -> None:
        if self._loader_token is not _LOADER_TOKEN:
            raise TypeError(
                "LoadedStaticGaugeHeadResponse is issued only by "
                "load_static_gauge_head_artifact")


def write_static_gauge_hall_artifact(
    path: str | Path,
    hall_transaction,
    *,
    mesh_xy: Mesh,
) -> None:
    """Collectively persist the sealed three-number Hall transaction."""
    from gw.qsgw_head import StaticGaugeHallTransaction

    if not isinstance(hall_transaction, StaticGaugeHallTransaction):
        raise TypeError(
            "StaticGaugeHall artifact requires the sealed canonical Hall "
            "transaction")
    stop = int(hall_transaction.band_stop)
    nk_tot = int(hall_transaction.nk_tot)
    wfn_sha256 = hall_transaction.wfn_fingerprint
    operator_sha256 = (
        hall_transaction.hamiltonian_config_operator_fingerprint)
    sigma_array = hall_transaction.sigma_H
    sigma_H = np.asarray(jax.device_get(sigma_array), dtype=np.float64)
    if sigma_H.shape != (3,) or not np.all(np.isfinite(sigma_H)):
        raise ValueError(
            "StaticGaugeHall sigma_H must be three finite real numbers")
    final_path, partial_path = _immutable_partial_paths(
        path, artifact_name="StaticGaugeHall")
    with SlabIO(str(partial_path), mode="w", mesh=mesh_xy) as io:
        io.write_attr(
            "schema_version", np.int32(STATIC_GAUGE_HALL_SCHEMA_VERSION))
        io.write_attr("complete", np.int32(1))
        io.write_attr(
            "wfn_fingerprint_i32",
            _text_i32(wfn_sha256, encoding="ascii"))
        io.write_attr(
            "hamiltonian_config_operator_fingerprint_i32",
            _text_i32(operator_sha256, encoding="ascii"))
        io.write_attr("band_stop", np.int32(stop))
        io.write_attr("nk_tot", np.int32(nk_tot))
        io.write_attr("sigma_H_cart", sigma_H)
        io.write_attr(
            "hall_antisymmetry_residual",
            np.float64(hall_transaction.antisymmetry_residual))

    _publish_completed_partial(
        partial_path, final_path, artifact_name="StaticGaugeHall",
        barrier_name="static_gauge_hall_artifact_published")


def load_static_gauge_hall_artifact(
    path: str | Path,
    *,
    mesh_xy: Mesh,
    wfn,
    expected_band_start: int,
    expected_band_stop: int,
    expected_nk_tot: int,
    wfn_fingerprint_binding=None,
):
    """Validate and load one immutable Hall transaction artifact."""
    from gw.qsgw_head import _static_gauge_hall_transaction_from_artifact

    artifact_path = Path(path)
    if artifact_path.name.endswith(".partial"):
        raise ValueError(
            "GATE static_gauge_hall_partial: refusing a partial artifact path")
    if not artifact_path.exists():
        raise FileNotFoundError(
            "GATE static_gauge_hall_artifact_absent: no completed artifact "
            f"exists at {artifact_path}")

    expected_wfn = _require_wfn_sha256(
        wfn_fingerprint(wfn)
        if wfn_fingerprint_binding is None
        else fingerprint_from_binding(wfn_fingerprint_binding, wfn))
    expected_start = int(expected_band_start)
    expected_stop = int(expected_band_stop)
    expected_nk = int(expected_nk_tot)
    if (expected_start != 0 or expected_stop <= expected_start
            or expected_stop > int(wfn.nbands) or expected_nk <= 0):
        raise ValueError(
            "expected StaticGaugeHall manifold must be bands [0,stop) "
            "within the WFN and nk_tot>0")

    with SlabIO(str(artifact_path), mode="r", mesh=mesh_xy) as io:
        complete = int(np.asarray(_read_required_small(io, "complete")))
        schema = int(np.asarray(_read_required_small(io, "schema_version")))
        artifact_wfn = _decode_i32_text(
            _read_required_small(io, "wfn_fingerprint_i32"),
            field_name="wfn_fingerprint_i32", encoding="ascii")
        operator_fingerprint = _decode_i32_text(
            _read_required_small(
                io, "hamiltonian_config_operator_fingerprint_i32"),
            field_name="hamiltonian_config_operator_fingerprint_i32",
            encoding="ascii")
        stop = int(np.asarray(_read_required_small(io, "band_stop")))
        nk_tot = int(np.asarray(_read_required_small(io, "nk_tot")))
        sigma_H = np.asarray(
            _read_required_small(io, "sigma_H_cart"), dtype=np.float64)
        hall_residual = float(np.asarray(_read_required_small(
            io, "hall_antisymmetry_residual")))

    if complete != 1:
        raise ValueError(
            f"StaticGaugeHall artifact is incomplete (complete={complete})")
    if schema != STATIC_GAUGE_HALL_SCHEMA_VERSION:
        raise ValueError(
            f"schema_version={schema}, expected "
            f"{STATIC_GAUGE_HALL_SCHEMA_VERSION}")
    artifact_wfn = _require_wfn_sha256(artifact_wfn)
    if artifact_wfn != expected_wfn:
        raise ValueError("StaticGaugeHall WFN identity differs")
    if stop != expected_stop:
        raise ValueError(
            f"StaticGaugeHall band_stop={stop}, expected {expected_stop}")
    if nk_tot != expected_nk:
        raise ValueError(
            f"StaticGaugeHall nk_tot={nk_tot}, expected {expected_nk}")
    if sigma_H.shape != (3,) or not np.all(np.isfinite(sigma_H)):
        raise ValueError(
            "StaticGaugeHall sigma_H_cart must contain three finite values")
    if not np.isfinite(hall_residual) or hall_residual < 0.0:
        raise ValueError("StaticGaugeHall antisymmetry residual is invalid")
    operator_fingerprint = _require_prefixed_sha256(
        operator_fingerprint,
        field_name="Hall Hamiltonian/config/operator fingerprint")

    return _static_gauge_hall_transaction_from_artifact(
        sigma_H=sigma_H,
        hamiltonian_config_operator_fingerprint=operator_fingerprint,
        wfn_fingerprint=artifact_wfn,
        band_start=expected_start,
        band_stop=stop,
        nk_tot=nk_tot,
        antisymmetry_residual=hall_residual,
        mesh=mesh_xy,
    )


def write_isometric_retained_bubble_source_artifact(
    path: str | Path,
    source,
    *,
    mesh_xy: Mesh,
) -> None:
    """Persist the bounded normalized source without q2/link/WFN replicas."""
    import json
    from gw.static_gauge_response import (
        require_isometric_retained_bubble_source)

    source = require_isometric_retained_bubble_source(source, mesh_xy)
    final_path, partial_path = _immutable_partial_paths(
        path, artifact_name="IsometricRetainedBubbleSource")
    logical = int(source.band_stop) - int(source.band_start)
    p_shape = (2, 4, int(source.nk_tot), logical, logical)
    availability_json = json.dumps(
        source.availability.as_tokens(), sort_keys=True, separators=(",", ":"))
    with SlabIO(str(partial_path), mode="w", mesh=mesh_xy) as io:
        io.create_dataset(
            _P_SOURCE_DATASET, shape=p_shape, dtype=np.complex128)
        io.create_dataset(
            _S_SOURCE_DATASET, shape=(2, 2, 4, 4), dtype=np.complex128)
        # SlabIO clips the producer's mesh padding to this logical dataset.
        # A consumer on a different mesh reads the same logical bytes into
        # its own padded carrier; processor count is not artifact semantics.
        io.write_slab(_P_SOURCE_DATASET, source.energy_scaled_d1_raw)
        io.write_slab(_S_SOURCE_DATASET, source.S_direct)
        io.write_attr(
            "schema_version",
            np.int32(ISOMETRIC_RETAINED_BUBBLE_SOURCE_SCHEMA_VERSION))
        io.write_attr("complete", np.int32(1))
        io.write_attr(
            "artifact_kind_i32",
            _text_i32("isometric_retained_bubble_source", encoding="ascii"))
        io.write_attr(
            "convention_id_i32",
            _text_i32(ISOMETRIC_RETAINED_BUBBLE_SOURCE_CONVENTION_ID))
        io.write_attr(
            "availability_json_i32", _text_i32(availability_json))
        for name, value in (
            ("charge_representation_i32", source.charge_representation),
            ("spatial_current_representation_i32",
             source.spatial_current_representation),
            ("endpoint_jet_convention_i32", source.endpoint_jet_convention),
            ("hamiltonian_config_operator_fingerprint_i32",
             source.hamiltonian_config_operator_fingerprint),
            ("source_fingerprint_i32", source.source_fingerprint),
            ("wfn_fingerprint_i32", source.wfn_fingerprint),
            ("approximation_i32", source.approximation),
        ):
            io.write_attr(name, _text_i32(value, encoding="ascii"))
        io.write_attr("band_start", np.int32(source.band_start))
        io.write_attr("band_stop", np.int32(source.band_stop))
        io.write_attr("nk_tot", np.int32(source.nk_tot))
        io.write_attr("band_extent", np.int32(logical))
        io.write_attr(
            "parallel_transport_schema_version",
            np.int32(source.parallel_transport_schema_version))
        io.write_attr(
            "parallel_transport_polar_rcond",
            np.float64(source.parallel_transport_polar_rcond))
        io.write_attr(
            "parallel_transport_derivative_axes",
            np.asarray(
                source.parallel_transport_derivative_axes, dtype=np.int32))
        io.write_attr(
            "parallel_transport_coefficient_frame_i32",
            _text_i32(
                source.parallel_transport_coefficient_frame,
                encoding="ascii"))
        io.write_attr("mixed_transition_tensor_cart", np.asarray(
            jax.device_get(source.mixed_transition_tensor),
            dtype=np.float64))
        for name in (
            "charge_ward_residual", "ward_residual", "hermiticity_residual",
            "ordered_curvature_residual", "q2_symmetry_residual",
            "hall_antisymmetry_residual",
        ):
            io.write_attr(name, np.float64(getattr(source, name)))

    _publish_completed_partial(
        partial_path, final_path,
        artifact_name="IsometricRetainedBubbleSource",
        barrier_name="isometric_retained_bubble_source_published")


def load_isometric_retained_bubble_source_artifact(
    path: str | Path,
    *,
    mesh_xy: Mesh,
    wfn,
    expected_band_start: int,
    expected_band_stop: int,
    expected_nk_tot: int,
    wfn_fingerprint_binding=None,
):
    """Load the immutable normalized source onto its native shardings."""
    import json
    from gw.static_gauge_response import (
        ISOMETRIC_ENDPOINT_JET_CONVENTION,
        ISOMETRIC_RETAINED_BUBBLE_APPROXIMATION,
        ISOMETRIC_RETAINED_BUBBLE_AVAILABILITY,
        _issue_isometric_retained_bubble_source,
        require_isometric_retained_bubble_source,
    )

    artifact_path = Path(path)
    if artifact_path.name.endswith(".partial"):
        raise ValueError(
            "GATE isometric_retained_bubble_partial: refusing a partial path")
    if not artifact_path.exists():
        raise FileNotFoundError(
            "GATE isometric_retained_bubble_artifact_absent: no completed "
            f"artifact exists at {artifact_path}")
    expected_wfn = _require_wfn_sha256(
        wfn_fingerprint(wfn)
        if wfn_fingerprint_binding is None
        else fingerprint_from_binding(wfn_fingerprint_binding, wfn))
    expected_start, expected_stop, expected_nk = (
        int(expected_band_start), int(expected_band_stop),
        int(expected_nk_tot))
    expected_availability = json.dumps(
        ISOMETRIC_RETAINED_BUBBLE_AVAILABILITY.as_tokens(),
        sort_keys=True, separators=(",", ":"))

    with SlabIO(str(artifact_path), mode="r", mesh=mesh_xy) as io:
        def text_field(name: str) -> str:
            return _decode_i32_text(
                _read_required_small(io, name), field_name=name,
                encoding="ascii")

        complete = int(np.asarray(_read_required_small(io, "complete")))
        schema = int(np.asarray(_read_required_small(io, "schema_version")))
        kind = text_field("artifact_kind_i32")
        convention = text_field("convention_id_i32")
        availability_json = text_field("availability_json_i32")
        charge_representation = text_field("charge_representation_i32")
        current_representation = text_field(
            "spatial_current_representation_i32")
        endpoint_convention = text_field("endpoint_jet_convention_i32")
        operator_fingerprint = text_field(
            "hamiltonian_config_operator_fingerprint_i32")
        source_fingerprint = text_field("source_fingerprint_i32")
        artifact_wfn = text_field("wfn_fingerprint_i32")
        approximation = text_field("approximation_i32")
        start = int(np.asarray(_read_required_small(io, "band_start")))
        stop = int(np.asarray(_read_required_small(io, "band_stop")))
        nk_tot = int(np.asarray(_read_required_small(io, "nk_tot")))
        band_extent = int(np.asarray(_read_required_small(io, "band_extent")))
        pt_schema = int(np.asarray(_read_required_small(
            io, "parallel_transport_schema_version")))
        pt_rcond = float(np.asarray(_read_required_small(
            io, "parallel_transport_polar_rcond")))
        pt_axes = tuple(int(axis) for axis in np.asarray(
            _read_required_small(io, "parallel_transport_derivative_axes"),
            dtype=np.int32).reshape(-1))
        pt_frame = text_field("parallel_transport_coefficient_frame_i32")
        mixed_transition_tensor = _read_required_small(
            io, "mixed_transition_tensor_cart")
        residuals = {name: float(np.asarray(_read_required_small(io, name)))
                     for name in (
                         "charge_ward_residual", "ward_residual",
                         "hermiticity_residual", "ordered_curvature_residual",
                         "q2_symmetry_residual",
                         "hall_antisymmetry_residual")}

        refusals = []
        if complete != 1:
            refusals.append(f"complete={complete}, expected 1")
        if schema != ISOMETRIC_RETAINED_BUBBLE_SOURCE_SCHEMA_VERSION:
            refusals.append(
                f"schema_version={schema}, expected "
                f"{ISOMETRIC_RETAINED_BUBBLE_SOURCE_SCHEMA_VERSION}")
        if kind != "isometric_retained_bubble_source":
            refusals.append(
                f"artifact kind {kind!r} is not a normalized response source")
        if convention != ISOMETRIC_RETAINED_BUBBLE_SOURCE_CONVENTION_ID:
            refusals.append("source Fourier/unit/payload convention differs")
        if availability_json != expected_availability:
            refusals.append("explicit term availability differs")
        if endpoint_convention != ISOMETRIC_ENDPOINT_JET_CONVENTION:
            refusals.append("endpoint-jet representation/convention differs")
        if approximation != ISOMETRIC_RETAINED_BUBBLE_APPROXIMATION:
            refusals.append("approximation declaration differs")
        if artifact_wfn != expected_wfn:
            refusals.append("WFN fingerprint differs")
        if (start, stop) != (expected_start, expected_stop):
            refusals.append(
                f"band interval [{start},{stop}) != expected "
                f"[{expected_start},{expected_stop})")
        if nk_tot != expected_nk:
            refusals.append(f"nk_tot={nk_tot}, expected {expected_nk}")
        if band_extent != stop - start:
            refusals.append(
                f"stored logical band extent {band_extent} != {stop-start}")
        if refusals:
            raise ValueError(
                "GATE isometric_retained_bubble_artifact_mismatch:\n  - "
                + "\n  - ".join(refusals))

        try:
            energy_scaled_d1_raw = io.read_slab(
                _P_SOURCE_DATASET,
                partition_spec=P(None, None, None, "x", "y"))
            S_direct = io.read_slab(
                _S_SOURCE_DATASET, shape=(2, 2, 4, 4), partition_spec=P())
        except (KeyError, RuntimeError, ValueError) as exc:
            raise ValueError(
                "GATE isometric_retained_bubble_schema: missing or unreadable "
                "P/S dataset") from exc

    from common.collectives import device_put_process_local
    from jax.sharding import NamedSharding
    mixed_transition_tensor = device_put_process_local(
        np.asarray(mixed_transition_tensor, dtype=np.float64),
        NamedSharding(mesh_xy, P()))
    source = _issue_isometric_retained_bubble_source(
        energy_scaled_d1_raw=energy_scaled_d1_raw,
        S_direct=S_direct,
        mixed_transition_tensor=mixed_transition_tensor,
        availability=ISOMETRIC_RETAINED_BUBBLE_AVAILABILITY,
        charge_representation=charge_representation,
        spatial_current_representation=current_representation,
        endpoint_jet_convention=endpoint_convention,
        hamiltonian_config_operator_fingerprint=operator_fingerprint,
        source_fingerprint=source_fingerprint,
        parallel_transport_schema_version=pt_schema,
        parallel_transport_polar_rcond=pt_rcond,
        parallel_transport_coefficient_frame=pt_frame,
        parallel_transport_derivative_axes=pt_axes,
        wfn_fingerprint=artifact_wfn,
        band_start=start, band_stop=stop, nk_tot=nk_tot,
        approximation=approximation,
        **residuals,
    )
    return require_isometric_retained_bubble_source(source, mesh_xy)


def write_static_gauge_head_artifact(
    path: str | Path,
    response: StaticGaugeHeadResponse,
    *,
    mesh_xy: Mesh,
    wfn,
    band_start: int,
    band_stop: int,
    body_response_fingerprint: str,
    source_write_ibz_only: bool,
    source_low_mem_bands: bool,
) -> None:
    """Collectively serialize and atomically publish one immutable response.

    This only serializes an already-produced response.  It computes the WFN
    identity through the canonical owner and records the band/body lineage;
    it does not construct a current, contact, Hall term, or response body.

    The final path is create-once.  SlabIO writes ``<path>.partial`` and
    stamps ``complete=1`` only after its sharded writes drain.  A hard-link
    publish then makes that completed inode visible at the final name without
    ever replacing an existing artifact.
    """
    response = require_static_gauge_head_response(response, mesh_xy)
    layout = response.layout
    if (layout.ordering != PHOTON_BASIS_ORDERING
            or layout.bare_propagator != PHOTON_BARE_PROPAGATOR):
        raise ValueError(
            "StaticGaugeHead artifact accepts only the canonical photon "
            "ordering and bare propagator")

    start, stop = int(band_start), int(band_stop)
    if start < 0 or stop <= start or stop > int(wfn.nbands):
        raise ValueError(
            "StaticGaugeHead band interval must satisfy "
            f"0 <= start < stop <= WFN.nbands; got [{start},{stop}) and "
            f"WFN.nbands={int(wfn.nbands)}")
    wfn_sha256 = _require_wfn_sha256(wfn_fingerprint(wfn))
    body_sha256 = _require_prefixed_sha256(
        body_response_fingerprint, field_name="body response fingerprint")
    operator_sha256 = _require_prefixed_sha256(
        response.hamiltonian_config_operator_fingerprint,
        field_name="Hamiltonian/config/operator fingerprint",
    )

    fixed_dtypes = (
        (response.S_direct, "S_direct", np.dtype(np.complex128)),
        (response.Y_x, "Y_x", np.dtype(np.complex128)),
        (response.Z_y, "Z_y", np.dtype(np.complex128)),
        (response.sigma_H, "sigma_H", np.dtype(np.float64)),
    )
    for array, name, wanted in fixed_dtypes:
        got = np.dtype(array.dtype)
        if got != wanted:
            raise TypeError(
                f"StaticGaugeHead {name} dtype {got} != fixed {wanted}")

    final_path, partial_path = _immutable_partial_paths(
        path, artifact_name="StaticGaugeHead")

    n_body = int(layout.packed_extent)
    with SlabIO(str(partial_path), mode="w", mesh=mesh_xy) as io:
        io.create_dataset(
            _S_DATASET, shape=(2, 2, 4, 4), dtype=np.complex128)
        io.create_dataset(
            _Y_DATASET, shape=(2, 4, n_body), dtype=np.complex128)
        io.create_dataset(
            _Z_DATASET, shape=(2, n_body, 4), dtype=np.complex128)
        io.write_slab(_S_DATASET, response.S_direct)
        io.write_slab(_Y_DATASET, response.Y_x)
        io.write_slab(_Z_DATASET, response.Z_y)
        io.write_attr(
            "schema_version", np.int32(STATIC_GAUGE_HEAD_SCHEMA_VERSION))
        io.write_attr("complete", np.int32(1))
        io.write_attr(
            "convention_id_i32", _text_i32(STATIC_GAUGE_HEAD_CONVENTION_ID))
        io.write_attr(
            "wfn_fingerprint_scheme_i32",
            _text_i32(WFN_FINGERPRINT_SCHEME, encoding="ascii"))
        io.write_attr(
            "wfn_fingerprint_i32", _text_i32(wfn_sha256, encoding="ascii"))
        io.write_attr(
            "hamiltonian_config_operator_fingerprint_i32",
            _text_i32(operator_sha256, encoding="ascii"))
        io.write_attr(
            "body_response_fingerprint_i32",
            _text_i32(body_sha256, encoding="ascii"))
        io.write_attr("band_start", np.int32(start))
        io.write_attr("band_stop", np.int32(stop))
        io.write_attr(
            "logical_extents",
            np.asarray(layout.logical_extents, dtype=np.int32))
        io.write_attr(
            "padded_extents",
            np.asarray(layout.padded_extents, dtype=np.int32))
        io.write_attr("mesh_side", np.int32(layout.mesh_side))
        io.write_attr(
            "sigma_H_cart", np.asarray(response.sigma_H, dtype=np.float64))
        io.write_attr("operator_current_equivalent", np.int32(1))
        io.write_attr("contact_is_exact", np.int32(1))
        io.write_attr("ward_residual", np.float64(response.ward_residual))
        io.write_attr(
            "hermiticity_residual",
            np.float64(response.hermiticity_residual))
        io.write_attr(
            "source_write_ibz_only", np.int32(source_write_ibz_only))
        io.write_attr(
            "source_low_mem_bands", np.int32(source_low_mem_bands))

    _publish_completed_partial(
        partial_path, final_path, artifact_name="StaticGaugeHead",
        barrier_name="static_gauge_head_artifact_published")


def load_static_gauge_head_artifact(
    path: str | Path,
    *,
    mesh_xy: Mesh,
    wfn,
    expected_band_start: int,
    expected_band_stop: int,
    expected_layout: PhotonBasisLayout,
    expected_body_response_fingerprint: str,
    expected_hamiltonian_config_operator_fingerprint: str,
) -> LoadedStaticGaugeHeadResponse:
    """Load one completed artifact directly onto its native shardings.

    All replicated lineage and layout fields are checked before either body
    wing is read.  ``Y`` returns on ``P(None,None,'x')`` and ``Z`` on
    ``P(None,'y',None)`` under both source storage policies.  Only O(1)
    metadata and the bounded Hall vector use :meth:`SlabIO.read_small`.
    """
    artifact_path = Path(path)
    if artifact_path.name.endswith(".partial"):
        raise ValueError(
            "GATE static_gauge_head_partial: refusing a partial artifact path")
    if not artifact_path.exists():
        raise FileNotFoundError(
            "GATE static_gauge_head_artifact_absent: no completed artifact "
            f"exists at {artifact_path}")

    expected_wfn = _require_wfn_sha256(wfn_fingerprint(wfn))
    expected_body = _require_prefixed_sha256(
        expected_body_response_fingerprint,
        field_name="expected body response fingerprint",
    )
    expected_operator = _require_prefixed_sha256(
        expected_hamiltonian_config_operator_fingerprint,
        field_name="expected Hamiltonian/config/operator fingerprint",
    )
    expected_start = int(expected_band_start)
    expected_stop = int(expected_band_stop)
    if (expected_start < 0 or expected_stop <= expected_start
            or expected_stop > int(wfn.nbands)):
        raise ValueError(
            "expected StaticGaugeHead band interval must satisfy "
            "0 <= start < stop <= WFN.nbands")
    if not isinstance(expected_layout, PhotonBasisLayout):
        raise TypeError(
            "expected_layout must be the canonical PhotonBasisLayout")
    expected_layout.assert_mesh(mesh_xy)

    with SlabIO(str(artifact_path), mode="r", mesh=mesh_xy) as io:
        complete = int(np.asarray(_read_required_small(io, "complete")))
        if complete != 1:
            raise ValueError(
                "GATE static_gauge_head_incomplete: artifact is not "
                f"complete (complete={complete})")

        schema = int(np.asarray(_read_required_small(io, "schema_version")))
        convention = _decode_i32_text(
            _read_required_small(io, "convention_id_i32"),
            field_name="convention_id_i32", encoding="utf-8")
        wfn_scheme = _decode_i32_text(
            _read_required_small(io, "wfn_fingerprint_scheme_i32"),
            field_name="wfn_fingerprint_scheme_i32", encoding="ascii")
        artifact_wfn = _decode_i32_text(
            _read_required_small(io, "wfn_fingerprint_i32"),
            field_name="wfn_fingerprint_i32", encoding="ascii")
        operator_fingerprint = _decode_i32_text(
            _read_required_small(
                io, "hamiltonian_config_operator_fingerprint_i32"),
            field_name="hamiltonian_config_operator_fingerprint_i32",
            encoding="ascii")
        body_fingerprint = _decode_i32_text(
            _read_required_small(io, "body_response_fingerprint_i32"),
            field_name="body_response_fingerprint_i32", encoding="ascii")
        start = int(np.asarray(_read_required_small(io, "band_start")))
        stop = int(np.asarray(_read_required_small(io, "band_stop")))
        logical = tuple(int(v) for v in np.asarray(
            _read_required_small(io, "logical_extents"),
            dtype=np.int32).reshape(-1))
        padded = tuple(int(v) for v in np.asarray(
            _read_required_small(io, "padded_extents"),
            dtype=np.int32).reshape(-1))
        mesh_side = int(np.asarray(_read_required_small(io, "mesh_side")))
        current_equivalent = int(np.asarray(_read_required_small(
            io, "operator_current_equivalent")))
        contact_exact = int(np.asarray(_read_required_small(
            io, "contact_is_exact")))
        write_ibz_only = int(np.asarray(_read_required_small(
            io, "source_write_ibz_only")))
        low_mem_bands = int(np.asarray(_read_required_small(
            io, "source_low_mem_bands")))
        ward_residual = float(np.asarray(_read_required_small(
            io, "ward_residual")))
        hermiticity_residual = float(np.asarray(_read_required_small(
            io, "hermiticity_residual")))
        sigma_H = _read_required_small(io, "sigma_H_cart")

        refusals = []
        if schema != STATIC_GAUGE_HEAD_SCHEMA_VERSION:
            refusals.append(
                f"schema_version={schema}, expected "
                f"{STATIC_GAUGE_HEAD_SCHEMA_VERSION}")
        if convention != STATIC_GAUGE_HEAD_CONVENTION_ID:
            refusals.append(
                "Fourier/sign/unit/normalization convention differs")
        if wfn_scheme != WFN_FINGERPRINT_SCHEME:
            refusals.append("WFN fingerprint scheme differs")
        if artifact_wfn != expected_wfn:
            refusals.append("WFN fingerprint differs")
        if (start, stop) != (expected_start, expected_stop):
            refusals.append(
                f"band interval [{start},{stop}) != expected "
                f"[{expected_start},{expected_stop})")
        if body_fingerprint != expected_body:
            refusals.append("packed body response fingerprint differs")
        if operator_fingerprint != expected_operator:
            refusals.append(
                "Hamiltonian/config/operator fingerprint differs")
        if current_equivalent != 1:
            refusals.append("operator_current_equivalent is not 1")
        if contact_exact != 1:
            refusals.append("contact_is_exact is not 1")
        if write_ibz_only not in (0, 1) or low_mem_bands not in (0, 1):
            refusals.append("storage-policy stamps are not zero or one")

        try:
            layout = PhotonBasisLayout(
                logical_extents=logical,
                padded_extents=padded,
                mesh_side=mesh_side,
            )
        except (TypeError, ValueError) as exc:
            refusals.append(f"invalid photon layout: {exc}")
            layout = None
        if layout is not None and layout != expected_layout:
            refusals.append(
                f"photon layout {layout!r} != expected {expected_layout!r}")
        if refusals:
            raise ValueError(
                "GATE static_gauge_head_artifact_mismatch: refusing "
                "StaticGaugeHead artifact:\n  - " + "\n  - ".join(refusals))

        n_body = int(expected_layout.packed_extent)
        try:
            S_direct = io.read_slab(
                _S_DATASET, shape=(2, 2, 4, 4), partition_spec=P())
            Y_x = io.read_slab(
                _Y_DATASET, shape=(2, 4, n_body),
                partition_spec=P(None, None, "x"))
            Z_y = io.read_slab(
                _Z_DATASET, shape=(2, n_body, 4),
                partition_spec=P(None, "y", None))
        except (KeyError, RuntimeError, ValueError) as exc:
            raise ValueError(
                "GATE static_gauge_head_schema: missing or unreadable "
                "S/Y/Z response dataset") from exc

    fixed_dtypes = (
        (S_direct, "S_direct", np.dtype(np.complex128)),
        (Y_x, "Y_x", np.dtype(np.complex128)),
        (Z_y, "Z_y", np.dtype(np.complex128)),
        (sigma_H, "sigma_H", np.dtype(np.float64)),
    )
    for array, name, wanted in fixed_dtypes:
        got = np.dtype(array.dtype)
        if got != wanted:
            raise TypeError(
                f"StaticGaugeHead {name} dtype {got} != fixed {wanted}")

    loaded = LoadedStaticGaugeHeadResponse(
        layout=expected_layout,
        S_direct=S_direct,
        sigma_H=sigma_H,
        Y_x=Y_x,
        Z_y=Z_y,
        hamiltonian_config_operator_fingerprint=operator_fingerprint,
        operator_current_equivalent=True,
        contact_is_exact=True,
        ward_residual=ward_residual,
        hermiticity_residual=hermiticity_residual,
        convention_id=convention,
        artifact_path=str(artifact_path),
        wfn_fingerprint_scheme=wfn_scheme,
        wfn_fingerprint=artifact_wfn,
        band_start=start,
        band_stop=stop,
        body_response_fingerprint=body_fingerprint,
        source_write_ibz_only=bool(write_ibz_only),
        source_low_mem_bands=bool(low_mem_bands),
        _loader_token=_LOADER_TOKEN,
    )
    require_static_gauge_head_response(loaded, mesh_xy)
    return loaded


__all__ = [
    "ISOMETRIC_RETAINED_BUBBLE_SOURCE_CONVENTION_ID",
    "ISOMETRIC_RETAINED_BUBBLE_SOURCE_SCHEMA_VERSION",
    "LoadedStaticGaugeHeadResponse",
    "STATIC_GAUGE_HALL_SCHEMA_VERSION",
    "STATIC_GAUGE_HEAD_CONVENTION_ID",
    "STATIC_GAUGE_HEAD_SCHEMA_VERSION",
    "load_isometric_retained_bubble_source_artifact",
    "load_static_gauge_hall_artifact",
    "load_static_gauge_head_artifact",
    "write_isometric_retained_bubble_source_artifact",
    "write_static_gauge_hall_artifact",
    "write_static_gauge_head_artifact",
]
