"""Immutable dual and raw-difference Pauli Fourier tables.

The canonical projection and overlap owners build these tables. Persistence
does not alter their quadrature, dual metric, spin harmonics or RKB carrier.
An explicit artifact is loaded or refused; loading never builds a replacement.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np

SCHEMA = "lorrax.atomic_fourier_cache.v1"
KINDS = ("projection", "delta")
ARRAY_KEYS = frozenset(("momentum", "radial", "ell", "kappa"))
RECEIPT_KEYS = frozenset(("source_identity", "maximum_absolute_error",
                          "maximum_scaled_error", "validation_points"))


def _json_bytes(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def _controls(control):
    supplied = {key: value for key, value in control.items() if key != "species_files"}
    allowed = {"momentum_max", "momentum_points", "relative_tolerance",
               "absolute_tolerance", "validation_points"}
    if set(supplied) - allowed:
        raise ValueError("unknown atomic Fourier cache controls")
    result = {"relative_tolerance": 1e-10, "absolute_tolerance": 1e-12,
              "validation_points": 64, **supplied}
    maximum, count = float(result["momentum_max"]), int(result["momentum_points"])
    checks = int(result["validation_points"])
    rel, absolute = float(result["relative_tolerance"]), float(result["absolute_tolerance"])
    if (not np.isfinite((maximum, rel, absolute)).all() or maximum <= 0
            or count != result["momentum_points"] or count < 8
            or checks != result["validation_points"] or checks < 4
            or rel < 0 or absolute < 0):
        raise ValueError("invalid atomic Fourier cache controls")
    _json_bytes(result)
    return result


def _owner_sources():
    result = {}
    for module in ("psp.augmentation_spinors", "psp.augmented_samples",
                   "psp.reconstruction_overlap", "psp.atomic_reconstruction",
                   "psp.atomic_fourier_cache", "common.bispinor_init",
                   "common.gamma_matrices"):
        result[module] = hashlib.sha256(Path(importlib.util.find_spec(module).origin).read_bytes()).hexdigest()
    return result


def atomic_fourier_cache_binding(data, control):
    """Bind both source tables, all atomic metadata and canonical owner files."""
    metadata = data["metadata"]
    if (metadata.get("operator_comparison", {}).get("authenticated") is not True
            or metadata.get("phase_branch_validated") is not True):
        raise ValueError("atomic Fourier cache requires authenticated atomic source and phase branch")
    return dict(schema=SCHEMA, representation="raw_pauli_fourier",
                roles={"projection": "local_pseudo_overlap_dual", "delta": "unscaled_ae_minus_ps"},
                owner_sources_sha256=_owner_sources(), controls=_controls(control),
                atomic_source_sha256=metadata["source_sha256"],
                atomic_payload_sha256=metadata["payload_sha256"],
                atomic_metadata_sha256=hashlib.sha256(_json_bytes(metadata)).hexdigest(),
                units={"momentum": "bohr^-1", "radial_fourier_amplitude": "bohr^3/2"},
                convention="integral exp(-i K.r) phi(r) d^3r; no cell volume, translation or RKB factor")


def _validate_caches(caches, data, control, required_momentum_max):
    from psp.augmented_samples import _projection_identity
    from psp.reconstruction_overlap import _delta_identity

    required = float(required_momentum_max)
    if not np.isfinite(required) or required < 0 or required > float(control["momentum_max"]):
        raise ValueError("required atomic Fourier momentum exceeds cached envelope")
    if set(caches) != set(KINDS):
        raise ValueError("atomic Fourier cache requires both projection and raw delta")
    momentum = np.linspace(0., float(control["momentum_max"]), int(control["momentum_points"]))
    cells = np.unique(np.linspace(0, len(momentum) - 2, int(control["validation_points"])).astype(int))
    identities = {"projection": _projection_identity(data), "delta": _delta_identity(data)}
    for kind in KINDS:
        cache = caches[kind]
        if set(cache) != ARRAY_KEYS | RECEIPT_KEYS:
            raise ValueError("atomic Fourier cache keys differ from schema")
        grid, l, k, values = (np.asarray(cache[key]) for key in ("momentum", "ell", "kappa", "radial"))
        if (grid.dtype != np.float64 or not np.array_equal(grid, momentum)
                or l.dtype.kind not in "iu" or k.dtype.kind not in "iu"
                or not np.array_equal(l, data["l"]) or not np.array_equal(k, data["kappa"])
                or l.ndim != 1 or k.shape != l.shape or not len(l)
                or np.any(l < 0) or np.any(k == 0) or np.any((k != l) & (k != -l - 1))):
            raise ValueError("atomic Fourier cache grid or labels differ from source/controls")
        if values.dtype != np.complex128 or values.shape != (len(grid), len(l)) or not np.isfinite(values).all():
            raise ValueError("atomic Fourier radial amplitudes must be finite complex128 on momentum/OPF grid")
        if cache["source_identity"] != identities[kind]:
            raise ValueError("atomic Fourier cache radial source identity mismatch")
        errors = (cache["maximum_absolute_error"], cache["maximum_scaled_error"])
        if (not np.isfinite(errors).all() or min(errors) < 0
                or cache["validation_points"] != len(cells)):
            raise ValueError("invalid atomic Fourier interpolation receipt")


def _pack(caches):
    return {f"{kind}_{key}": np.asarray(caches[kind][key]) for kind in KINDS for key in ARRAY_KEYS}


def _payload_hash(arrays):
    digest = hashlib.sha256()
    for name, value in sorted(arrays.items()):
        digest.update(name.encode())
        digest.update(str(value.shape).encode())
        digest.update(value.dtype.str.encode())
        digest.update(value.tobytes())
    return digest.hexdigest()


def build_atomic_fourier_caches(data, control):
    """Call the existing dual and raw-delta factories once per species."""
    from psp.augmented_samples import build_projection_radial_cache
    from psp.reconstruction_overlap import build_delta_radial_cache

    binding = atomic_fourier_cache_binding(data, control)
    controls = binding["controls"]
    caches = dict(projection=build_projection_radial_cache(data, **controls),
                  delta=build_delta_radial_cache(data, **controls))
    _validate_caches(caches, data, controls, 0.)
    return caches


def write_atomic_fourier_caches(path, caches, data, control):
    """Write a new immutable artifact; preserve every existing pathname."""
    binding = atomic_fourier_cache_binding(data, control)
    _validate_caches(caches, data, binding["controls"], 0.)
    arrays = _pack(caches)
    metadata = dict(binding=binding, payload_sha256=_payload_hash(arrays),
                    interpolation={kind: {key: caches[kind][key] for key in RECEIPT_KEYS} for kind in KINDS})
    metadata["metadata_sha256"] = hashlib.sha256(_json_bytes(metadata)).hexdigest()
    with Path(path).open("xb") as handle:
        np.savez_compressed(handle, **arrays, metadata_json=np.asarray(_json_bytes(metadata).decode()))
    return metadata


def load_atomic_fourier_caches(path, data, control, *, required_momentum_max=0.):
    """Strictly authenticate both tables, including requested momentum coverage."""
    with np.load(path, allow_pickle=False) as archive:
        metadata = json.loads(str(archive["metadata_json"]))
        arrays = {key: archive[key] for key in archive.files if key != "metadata_json"}
    checksum = metadata.pop("metadata_sha256", None)
    if checksum != hashlib.sha256(_json_bytes(metadata)).hexdigest():
        raise ValueError("atomic Fourier cache metadata checksum mismatch")
    expected = atomic_fourier_cache_binding(data, control)
    if metadata.get("binding") != expected:
        raise ValueError("atomic Fourier cache source, metadata, controls or owner provenance mismatch")
    if (set(arrays) != {f"{kind}_{key}" for kind in KINDS for key in ARRAY_KEYS}
            or set(metadata.get("interpolation", {})) != set(KINDS)
            or any(set(metadata["interpolation"][kind]) != RECEIPT_KEYS for kind in KINDS)):
        raise ValueError("atomic Fourier cache payload keys differ from schema")
    if metadata.get("payload_sha256") != _payload_hash(arrays):
        raise ValueError("atomic Fourier cache array payload checksum mismatch")
    caches = {kind: dict(metadata["interpolation"][kind],
                        **{key: arrays[f"{kind}_{key}"] for key in ARRAY_KEYS}) for kind in KINDS}
    _validate_caches(caches, data, expected["controls"], required_momentum_max)
    return caches
