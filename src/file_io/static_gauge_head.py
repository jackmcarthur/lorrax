"""Immutable SlabIO format of Hall frequency samples (``static_gauge_hall.h5``).

This is the sole format owner for the frequency-indexed Hall pseudovector
``sigma_H(z)`` produced by :func:`gw.qsgw_head.static_gauge_hall_transaction`
(``get_dipole_mtxels --static-gauge-hall-only``) and consumed, optionally, by
the packed photon Gamma-cell completion
(:func:`gw.w_isdf.compute_static_photon_response` under
the static or dynamic supported envelope). Schema v3 contains the 32-point
nested imaginary-axis Faraday MPA oracle, three complex components per sample,
and the exact plan identity; schema v2 remains readable for its static row.
The artifact is small and replicated; SlabIO is still the only transport so
that every rank reads the same completed inode collectively.

The loader authenticates the artifact against the consuming run: the WFN
identity (:func:`common.parallel_transport.wfn_fingerprint`), the band
manifold ``[0, stop)`` and the full-BZ k-count.  A present-but-mismatched
file refuses; it never degrades to ``sigma_H = 0``.  The absent-file default
(``sigma_H = 0``) is decided by the consumer, not here.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import jax
import numpy as np
from jax.sharding import Mesh

from common.collectives import barrier
from common.parallel_transport import (
    fingerprint_from_binding,
    wfn_fingerprint,
)
from file_io.slab_io import SlabIO


STATIC_GAUGE_HALL_SCHEMA_VERSION = 3
STATIC_GAUGE_HALL_READABLE_SCHEMA_VERSIONS = (2, 3)


def _sample_plan_sha256(
    frequencies_ry,
    *,
    label: str,
    n_poles: int,
    alpha: int,
    schedule: str,
    omega_max_ry: float,
) -> str:
    """Hash the exact frequency bytes and their MPA-plan provenance."""
    frequencies = np.ascontiguousarray(
        np.asarray(frequencies_ry, dtype=np.complex128))
    payload = json.dumps(
        {
            "label": str(label),
            "n_poles": int(n_poles),
            "sampling_alpha": int(alpha),
            "sampling_schedule": str(schedule),
            "omega_max_ry": float(omega_max_ry),
        },
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    digest = hashlib.sha256()
    digest.update(payload)
    digest.update(frequencies.view(np.uint8))
    return "sha256:" + digest.hexdigest()


def _require_prefixed_sha256(value: str, *, field_name: str) -> str:
    value = str(value).strip()
    if (not value.startswith("sha256:")
            or len(value) != len("sha256:") + 64
            or any(c not in "0123456789abcdef" for c in value[7:])):
        raise ValueError(
            f"StaticGaugeHall {field_name} must be "
            "sha256:<64 lowercase hex>")
    return value


def _require_wfn_sha256(value: str) -> str:
    value = str(value).strip()
    if (len(value) != 64
            or any(c not in "0123456789abcdef" for c in value)):
        raise ValueError(
            "canonical StaticGaugeHall WFN fingerprint must be 64 "
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
            "GATE static_gauge_hall_schema: a required artifact field is "
            "unreadable.\n"
            f"  got:  StaticGaugeHall field {name!r}: "
            f"{type(exc).__name__}: {exc}\n"
            f"  want: a readable {name!r} field in a complete schema-v"
            f"{STATIC_GAUGE_HALL_SCHEMA_VERSION} artifact\n"
            "  why:  the Hall value is accepted only with its WFN, band, "
            "operator, and k-count provenance; a missing field makes that "
            "authentication impossible\n"
            "  doc:  docs/input_reference.md, static_gauge_hall_file.") from exc


def _decode_i32_text(value, *, field_name: str, encoding: str) -> str:
    raw = np.asarray(value, dtype=np.int32).reshape(-1)
    if np.any(raw < 0) or np.any(raw > 255):
        raise ValueError(
            "GATE static_gauge_hall_schema: an encoded text field contains "
            "a non-byte value.\n"
            f"  got:  {field_name} value range "
            f"[{int(raw.min())}, {int(raw.max())}]\n"
            f"  want: every {field_name} value in [0, 255]\n"
            "  why:  values outside one byte cannot encode the provenance "
            "text needed to authenticate this artifact\n"
            "  doc:  docs/input_reference.md, static_gauge_hall_file.")
    try:
        return raw.astype(np.uint8).tobytes().decode(encoding)
    except UnicodeDecodeError as exc:
        raise ValueError(
            "GATE static_gauge_hall_schema: an encoded provenance field "
            "cannot be decoded.\n"
            f"  got:  {field_name} is not valid {encoding}: {exc}\n"
            f"  want: {field_name} encoded as valid {encoding}\n"
            "  why:  undecodable provenance cannot authenticate the Hall "
            "artifact against the consuming calculation\n"
            "  doc:  docs/input_reference.md, static_gauge_hall_file.") from exc


def _immutable_partial_paths(
    path: str | Path, *, artifact_name: str,
) -> tuple[Path, Path]:
    """Validate one create-once destination."""
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


def write_static_gauge_hall_artifact(
    path: str | Path,
    hall_transaction,
    *,
    mesh_xy: Mesh,
) -> None:
    """Collectively persist the sealed Hall frequency transaction.

    The final path is create-once.  SlabIO writes ``<path>.partial`` and
    stamps ``complete=1`` only after its writes drain; a hard-link publish
    then makes that completed inode visible at the final name without ever
    replacing an existing artifact.
    """
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
    frequencies = np.asarray(
        jax.device_get(hall_transaction.frequencies_ry), dtype=np.complex128)
    sigma_H = np.asarray(
        jax.device_get(hall_transaction.sigma_H_frequency),
        dtype=np.complex128)
    schema_version = int(hall_transaction.artifact_schema_version)
    if (frequencies.ndim != 1 or frequencies.size == 0
            or not np.all(np.isfinite(frequencies))):
        raise ValueError(
            "StaticGaugeHall frequencies_ry must be a nonempty finite vector")
    if (sigma_H.shape != (frequencies.size, 3)
            or not np.all(np.isfinite(sigma_H))):
        raise ValueError(
            "StaticGaugeHall sigma_H_frequency must contain three finite "
            "values per frequency")
    final_path, partial_path = _immutable_partial_paths(
        path, artifact_name="StaticGaugeHall")
    with SlabIO(str(partial_path), mode="w", mesh=mesh_xy) as io:
        io.write_attr("schema_version", np.int32(schema_version))
        io.write_attr("complete", np.int32(1))
        io.write_attr(
            "wfn_fingerprint_i32",
            _text_i32(wfn_sha256, encoding="ascii"))
        io.write_attr(
            "hamiltonian_config_operator_fingerprint_i32",
            _text_i32(operator_sha256, encoding="ascii"))
        io.write_attr("band_stop", np.int32(stop))
        io.write_attr("nk_tot", np.int32(nk_tot))
        io.write_attr("frequency_ry", frequencies)
        io.write_attr("sigma_H_cart_frequency", sigma_H)
        if schema_version == 3:
            plan_hash = _sample_plan_sha256(
                frequencies,
                label=hall_transaction.sample_plan_label,
                n_poles=hall_transaction.sample_plan_n_poles,
                alpha=hall_transaction.sample_plan_alpha,
                schedule=hall_transaction.sample_plan_schedule,
                omega_max_ry=hall_transaction.sample_plan_omega_max_ry,
            )
            io.write_attr(
                "sample_plan_label_i32",
                _text_i32(hall_transaction.sample_plan_label))
            io.write_attr(
                "sample_plan_n_poles",
                np.int32(hall_transaction.sample_plan_n_poles))
            io.write_attr(
                "sample_plan_alpha",
                np.int32(hall_transaction.sample_plan_alpha))
            io.write_attr(
                "sample_plan_schedule_i32",
                _text_i32(hall_transaction.sample_plan_schedule))
            io.write_attr(
                "sample_plan_omega_max_ry",
                np.float64(hall_transaction.sample_plan_omega_max_ry))
            io.write_attr(
                "sample_plan_sha256_i32",
                _text_i32(plan_hash, encoding="ascii"))

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
    """Validate and load one immutable Hall transaction artifact.

    Returns the sealed :class:`gw.qsgw_head.StaticGaugeHallTransaction`
    placed on the run mesh.  Every identity mismatch refuses by name.
    """
    from gw.qsgw_head import _static_gauge_hall_transaction_from_artifact

    artifact_path = Path(path)
    if artifact_path.name.endswith(".partial"):
        raise ValueError(
            "GATE static_gauge_hall_partial: static_gauge_hall_file names "
            "an unpublished partial artifact.\n"
            f"  got:  static_gauge_hall_file = {artifact_path}\n"
            "  want: a completed artifact path without the .partial suffix\n"
            "  why:  .partial is the writer's in-flight inode and has not "
            "passed the collective completion/publication barrier\n"
            "  doc:  docs/input_reference.md, static_gauge_hall_file.")
    if not artifact_path.exists():
        raise FileNotFoundError(
            "GATE static_gauge_hall_file_missing: static_gauge_hall_file "
            "names no completed artifact.\n"
            f"  got:  static_gauge_hall_file = {artifact_path} (absent)\n"
            "  want: the completed artifact produced by "
            "get_dipole_mtxels --static-gauge-hall-only, or an unnamed key "
            "for sigma_H = 0\n"
            "  why:  treating a mistyped or missing named file as the "
            "unnamed zero-Hall default would silently change the model\n"
            "  doc:  docs/input_reference.md, static_gauge_hall_file.")

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
        frequencies = np.asarray(
            _read_required_small(io, "frequency_ry"), dtype=np.complex128)
        sigma_H = np.asarray(
            _read_required_small(io, "sigma_H_cart_frequency"),
            dtype=np.complex128)
        if schema == 3:
            sample_plan_label = _decode_i32_text(
                _read_required_small(io, "sample_plan_label_i32"),
                field_name="sample_plan_label_i32", encoding="utf-8")
            sample_plan_n_poles = int(np.asarray(
                _read_required_small(io, "sample_plan_n_poles")))
            sample_plan_alpha = int(np.asarray(
                _read_required_small(io, "sample_plan_alpha")))
            sample_plan_schedule = _decode_i32_text(
                _read_required_small(io, "sample_plan_schedule_i32"),
                field_name="sample_plan_schedule_i32", encoding="utf-8")
            sample_plan_omega_max_ry = float(np.asarray(
                _read_required_small(io, "sample_plan_omega_max_ry")))
            sample_plan_hash = _decode_i32_text(
                _read_required_small(io, "sample_plan_sha256_i32"),
                field_name="sample_plan_sha256_i32", encoding="ascii")
        else:
            sample_plan_label = None
            sample_plan_n_poles = None
            sample_plan_alpha = None
            sample_plan_schedule = None
            sample_plan_omega_max_ry = None
            sample_plan_hash = None

    if complete != 1:
        raise ValueError(
            f"StaticGaugeHall artifact is incomplete (complete={complete})")
    if schema not in STATIC_GAUGE_HALL_READABLE_SCHEMA_VERSIONS:
        raise ValueError(
            f"schema_version={schema}, expected one of "
            f"{STATIC_GAUGE_HALL_READABLE_SCHEMA_VERSIONS}")
    artifact_wfn = _require_wfn_sha256(artifact_wfn)
    if artifact_wfn != expected_wfn:
        raise ValueError("StaticGaugeHall WFN identity differs")
    if stop != expected_stop:
        raise ValueError(
            f"StaticGaugeHall band_stop={stop}, expected {expected_stop}")
    if nk_tot != expected_nk:
        raise ValueError(
            f"StaticGaugeHall nk_tot={nk_tot}, expected {expected_nk}")
    if (frequencies.ndim != 1 or frequencies.size == 0
            or not np.all(np.isfinite(frequencies))):
        raise ValueError(
            "StaticGaugeHall frequency_ry must be a nonempty finite vector")
    if (sigma_H.shape != (frequencies.size, 3)
            or not np.all(np.isfinite(sigma_H))):
        raise ValueError(
            "StaticGaugeHall sigma_H_cart_frequency must contain three "
            "finite values per frequency")
    if schema == 3:
        expected_plan_hash = _sample_plan_sha256(
            frequencies,
            label=sample_plan_label,
            n_poles=sample_plan_n_poles,
            alpha=sample_plan_alpha,
            schedule=sample_plan_schedule,
            omega_max_ry=sample_plan_omega_max_ry,
        )
        sample_plan_hash = _require_prefixed_sha256(
            sample_plan_hash, field_name="Hall MPA sample-plan fingerprint")
        if sample_plan_hash != expected_plan_hash:
            raise ValueError(
                "GATE static_gauge_hall_sample_plan_provenance: schema-v3 "
                "sample frequencies differ from their MPA-plan stamp")
    operator_fingerprint = _require_prefixed_sha256(
        operator_fingerprint,
        field_name="Hall Hamiltonian/config/operator fingerprint")

    return _static_gauge_hall_transaction_from_artifact(
        frequencies_ry=frequencies,
        sigma_H_frequency=sigma_H,
        hamiltonian_config_operator_fingerprint=operator_fingerprint,
        wfn_fingerprint=artifact_wfn,
        band_start=expected_start,
        band_stop=stop,
        nk_tot=nk_tot,
        mesh=mesh_xy,
        artifact_schema_version=schema,
        sample_plan_label=sample_plan_label,
        sample_plan_n_poles=sample_plan_n_poles,
        sample_plan_alpha=sample_plan_alpha,
        sample_plan_schedule=sample_plan_schedule,
        sample_plan_omega_max_ry=sample_plan_omega_max_ry,
    )


__all__ = [
    "STATIC_GAUGE_HALL_SCHEMA_VERSION",
    "STATIC_GAUGE_HALL_READABLE_SCHEMA_VERSIONS",
    "load_static_gauge_hall_artifact",
    "write_static_gauge_hall_artifact",
]
