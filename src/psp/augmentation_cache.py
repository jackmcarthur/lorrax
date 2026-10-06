"""Immutable normalized-RKB radial caches shared by generation and fitting.

The Hankel factory is the former fitting-stage implementation, with identical
quadratures. Persisting its result changes preparation cost, not the carrier.
All lengths are bohr and the source radial functions use normalized u=rR.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np

SCHEMA = "lorrax.normalized_augmentation_cache.v1"
ARRAY_KEYS = frozenset(("radius", "ell", "kappa", "large_R", "dlarge_R_dr",
                        "small_R", "dsmall_R_dr"))
RADIAL_KEYS = ("large_R", "dlarge_R_dr", "small_R", "dsmall_R_dr")


def _json_bytes(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def _controls(control, support_radius):
    result = {key: value for key, value in control.items() if key != "species_files"}
    km, nk = float(result["momentum_max"]), int(result["momentum_points"])
    rm, nr = float(result["radius_max"]), int(result["radius_points"])
    if (not np.isfinite((km, rm, support_radius)).all() or km <= 0 or nk < 4
            or rm <= support_radius or nr < 8 or support_radius <= 0
            or nk != result["momentum_points"] or nr != result["radius_points"]):
        raise ValueError("normalized cache requires positive resolved momentum and radii beyond support")
    if result.get("momentum_quadrature", "midpoint") not in ("midpoint", "gauss_legendre"):
        raise ValueError("unknown normalized-cache momentum quadrature")
    kind = result.get("radius_kind", "linear")
    if kind not in ("linear", "log"):
        raise ValueError("unknown normalized-cache radius grid")
    if kind == "log" and not 0 < float(result["radius_min"]) < rm:
        raise ValueError("normalized cache requires 0 < radius_min < radius_max")
    if "source_quadrature_order" in result:
        order = int(result["source_quadrature_order"])
        if order < 2 or order != result["source_quadrature_order"]:
            raise ValueError("normalized cache source quadrature order must be integer >=2")
    _json_bytes(result)
    return result


def _radius_grid(control):
    rm, nr = float(control["radius_max"]), int(control["radius_points"])
    if control.get("radius_kind", "linear") == "log":
        return np.concatenate(([0.], np.geomspace(float(control["radius_min"]), rm, nr - 1)))
    return np.linspace(0., rm, nr)


def normalized_cache_binding(data, control, *, support_radius):
    """Bind atomic payload/metadata, numerical controls and carrier sources."""
    from common.bispinor_init import NORMALIZED_RKB_LIFT_PROVENANCE, HALFALPHA

    metadata = data["metadata"]
    if (metadata.get("operator_comparison", {}).get("authenticated") is not True
            or metadata.get("phase_branch_validated") is not True):
        raise ValueError("normalized cache requires authenticated atomic source and phase branch")
    owners = {}
    for module in ("common.bispinor_init", "common.gamma_matrices", "psp.augmentation_spinors",
                   "psp.atomic_reconstruction", "psp.augmentation_cache"):
        origin = importlib.util.find_spec(module).origin
        owners[module] = hashlib.sha256(Path(origin).read_bytes()).hexdigest()
    return {"schema": SCHEMA, "carrier": "normalized_rkb",
            "carrier_provenance": NORMALIZED_RKB_LIFT_PROVENANCE,
            "half_alpha_fs": float(HALFALPHA), "owner_sources_sha256": owners,
            "atomic_source_sha256": metadata["source_sha256"],
            "atomic_payload_sha256": metadata["payload_sha256"],
            "atomic_metadata_sha256": hashlib.sha256(_json_bytes(metadata)).hexdigest(),
            "controls": _controls(control, support_radius), "support_radius": float(support_radius),
            "units": {"radius": "bohr", "momentum": "bohr^-1",
                      "radial_wavefunction": "bohr^-3/2", "radial_derivative": "bohr^-5/2"}}


def _validate_arrays(cache, data, control):
    if set(cache) != ARRAY_KEYS:
        raise ValueError("normalized cache array keys differ from schema")
    r, l, k = (np.asarray(cache[name]) for name in ("radius", "ell", "kappa"))
    if (r.dtype != np.float64 or not np.array_equal(r, _radius_grid(control))
            or not np.array_equal(l, data["l"]) or not np.array_equal(k, data["kappa"])
            or l.dtype.kind not in "iu" or k.dtype.kind not in "iu"
            or l.ndim != 1 or k.shape != l.shape or not len(l)
            or np.any(l < 0) or np.any(k == 0) or np.any((k != l) & (k != -l - 1))):
        raise ValueError("normalized cache radial grid or atomic labels differ from source/controls")
    for name in RADIAL_KEYS:
        value = np.asarray(cache[name])
        if value.shape != (len(r), len(l)) or value.dtype != np.complex128 or not np.isfinite(value).all():
            raise ValueError(f"normalized cache {name} must be finite complex128 on the radial/OPF grid")


def _payload_hash(cache):
    digest = hashlib.sha256()
    for name, value in sorted(cache.items()):
        value = np.asarray(value)
        digest.update(name.encode())
        digest.update(str(value.shape).encode())
        digest.update(value.dtype.str.encode())
        digest.update(value.tobytes())
    return digest.hexdigest()


def build_normalized_cache(data, control, *, support_radius):
    """Build the unchanged fitting-stage Hankel cache exactly once."""
    from scipy.special import roots_legendre
    from psp.atomic_reconstruction import evaluate_radial_correction
    from psp.augmentation_spinors import build_normalized_radial_cache

    control = _controls(control, support_radius)
    if float(data["r"][-1]) > support_radius:
        raise ValueError("atomic pre-lift reconstruction radius exceeds compact support")
    km, nk = float(control["momentum_max"]), int(control["momentum_points"])
    if control.get("momentum_quadrature", "midpoint") == "gauss_legendre":
        nodes, weights = roots_legendre(nk)
        momentum, dk_weights = .5*km*(nodes + 1), .5*km*weights
    else:
        dk = km/nk
        momentum, dk_weights = (np.arange(nk) + .5)*dk, np.full(nk, dk)
    source_r, source_w, source_delta = data["r"], data["weights_dr"], data["delta_R"]
    if "source_quadrature_order" in control:
        nodes, weights = roots_legendre(int(control["source_quadrature_order"]))
        midpoint, halfwidth = .5*(source_r[1:] + source_r[:-1]), .5*np.diff(source_r)
        source_r = (midpoint[:, None] + halfwidth[:, None]*nodes).reshape(-1)
        source_w = (halfwidth[:, None]*weights).reshape(-1)
        source_delta = evaluate_radial_correction(data, source_r)[0]
    cache = build_normalized_radial_cache(source_delta, source_r, source_w,
        data["l"], data["kappa"], momentum, dk_weights, _radius_grid(control))
    _validate_arrays(cache, data, control)
    return cache


def normalized_cache_tail_diagnostics(cache, *, support_radius):
    """Same radial tail estimates as the fitting stage; no accuracy claim."""
    radius = cache["radius"]
    outside = radius >= support_radius
    l, k = cache["ell"], cache["kappa"]
    small_l = 2*np.abs(k) - 1 - l
    density = radius[:, None]**2*(abs(cache["large_R"])**2 + abs(cache["small_R"])**2)
    gradient = radius[:, None]**2*(abs(cache["dlarge_R_dr"])**2 + abs(cache["dsmall_R_dr"])**2)
    gradient += l*(l + 1)*abs(cache["large_R"])**2 + small_l*(small_l + 1)*abs(cache["small_R"])**2
    result = {}
    for name, integrand in (("norm", density), ("gradient", gradient)):
        total = np.trapezoid(integrand, radius, axis=0)
        tail = np.trapezoid(integrand[outside], radius[outside], axis=0)
        ratio = np.divide(tail, total, out=np.zeros_like(total), where=total > 0)
        result[name] = float(np.max(ratio))
    return result


def write_normalized_cache(path, cache, data, control, *, support_radius):
    """Write a new immutable artifact, preserving every existing file."""
    path = Path(path)
    if path.exists():
        raise FileExistsError(f"preserve existing normalized cache: {path}")
    binding = normalized_cache_binding(data, control, support_radius=support_radius)
    _validate_arrays(cache, data, binding["controls"])
    metadata = {"binding": binding, "payload_sha256": _payload_hash(cache)}
    metadata["metadata_sha256"] = hashlib.sha256(_json_bytes(metadata)).hexdigest()
    np.savez_compressed(path, **cache, metadata_json=np.asarray(_json_bytes(metadata).decode()))
    return metadata


def load_normalized_cache(path, data, control, *, support_radius):
    """Load or refuse; an explicit artifact never triggers a hidden rebuild."""
    with np.load(path, allow_pickle=False) as archive:
        metadata = json.loads(str(archive["metadata_json"]))
        cache = {key: archive[key] for key in archive.files if key != "metadata_json"}
    checksum = metadata.pop("metadata_sha256", None)
    if checksum != hashlib.sha256(_json_bytes(metadata)).hexdigest():
        raise ValueError("normalized cache metadata checksum mismatch")
    expected = normalized_cache_binding(data, control, support_radius=support_radius)
    if metadata.get("binding") != expected:
        raise ValueError("normalized cache source, metadata, controls or carrier provenance mismatch")
    _validate_arrays(cache, data, expected["controls"])
    if metadata.get("payload_sha256") != _payload_hash(cache):
        raise ValueError("normalized cache array payload checksum mismatch")
    return cache
