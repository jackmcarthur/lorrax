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

import numpy as np
from jax.sharding import Mesh, PartitionSpec as P

from common.collectives import barrier
from common.parallel_transport import WFN_FINGERPRINT_SCHEME, wfn_fingerprint
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


STATIC_GAUGE_HEAD_SCHEMA_VERSION = 2
STATIC_GAUGE_HEAD_CONVENTION_ID = (
    "lorrax.static_gauge_head/v2"
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
    "|hall_CT=+i*epsilon[b,a,i]*sigma_H[b]*q[a]"
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

_S_DATASET = "S_direct_cart"
_Y_DATASET = "Y_cart_x"
_Z_DATASET = "Z_cart_y"
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

    final_path = Path(path)
    partial_path = Path(str(final_path) + ".partial")
    if final_path.name.endswith(".partial"):
        raise ValueError("StaticGaugeHead final path may not end in '.partial'")
    if not final_path.parent.is_dir():
        raise FileNotFoundError(
            "StaticGaugeHead parent directory does not exist: "
            f"{final_path.parent}")
    if os.path.lexists(final_path):
        raise FileExistsError(
            f"immutable StaticGaugeHead artifact already exists: {final_path}")
    if os.path.lexists(partial_path):
        raise FileExistsError(
            f"stale/in-flight StaticGaugeHead partial exists: {partial_path}")

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

    try:
        os.link(partial_path, final_path)
    except FileExistsError:
        if not os.path.samefile(partial_path, final_path):
            raise FileExistsError(
                "immutable StaticGaugeHead publish collided with a different "
                f"artifact at {final_path}") from None
    barrier("static_gauge_head_artifact_published")
    try:
        os.unlink(partial_path)
    except FileNotFoundError:
        pass


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
    "LoadedStaticGaugeHeadResponse",
    "STATIC_GAUGE_HEAD_CONVENTION_ID",
    "STATIC_GAUGE_HEAD_SCHEMA_VERSION",
    "load_static_gauge_head_artifact",
    "write_static_gauge_head_artifact",
]
