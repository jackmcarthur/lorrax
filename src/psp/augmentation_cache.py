"""Immutable normalized-RKB upper caches and their compact graph descriptor.

The stored Hankel arrays retain the original unwindowed transform evidence.
Production fields taper the large Hermite polynomial and derive the small
block from its gradient. All lengths are bohr; atomic source waves use u=rR.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np

SCHEMA = "lorrax.normalized_augmentation_cache.v2"
ARRAY_KEYS = frozenset(("radius", "ell", "kappa", "large_R", "dlarge_R_dr",
                        "small_R", "dsmall_R_dr", "field_model", "taper_start",
                        "support_radius", "half_alpha"))
RADIAL_KEYS = ("large_R", "dlarge_R_dr", "small_R", "dsmall_R_dr")
PAIRED_COMPACT_PAULI_FIELD_POLICY = 'unwindowed_U_of_compact_native_pauli'
PAIRED_AE_LARGE_FIELD_POLICY = 'paired_ae_large_preserved_free_graph'
NATIVE_PAULI_TARGET = "native_large_as_pauli"
AE_LARGE_TARGET = "ae_large_preserved_free_graph"
_TARGET_KEYS = frozenset(("nuclear_charge", "matched_dirac_file", "matched_dirac_sha256",
    "pseudo_exterior_file", "pseudo_exterior_sha256", "source_upf_file", "source_upf_sha256",
    "dirac_window_start", "dirac_window_stop", "completion_start", "completion_stop"))


def paired_field_policy_contract(policy):
    """Resolve one closed paired-field/frame contract, never infer its target.

    Native fields retain their original compact-Pauli target. The distinct
    AE-large target uses the compact construction large/free-small norm;
    finite-K represented fields and their tails are separate evidence.
    """
    if policy == PAIRED_COMPACT_PAULI_FIELD_POLICY:
        return dict(field_policy=policy,
            field_schema='lorrax.dev.common_chi_species_fields.v1',
            field_source_model='compact_native_Pauli_target_with_finiteK_paired_approximation',
            frame_schema='lorrax.dev.compact_pauli_common_frame.v1',
            frame_model='compact_native_pauli_common_frame_v1',
            overlap_operator='compact_native_pauli_target',
            source_frame_policy='compact_native_pauli_common_A_before_U')
    if policy == PAIRED_AE_LARGE_FIELD_POLICY:
        return dict(field_policy=policy,
            field_schema='lorrax.dev.paired_ae_large_species_fields.v1',
            field_source_model='ae_large_preserved_free_graph_with_finiteK_paired_approximation',
            frame_schema='lorrax.dev.ae_large_paired_common_frame.v1',
            frame_model='ae_large_preserved_free_graph_common_frame_v1',
            overlap_operator='ae_large_preserved_free_graph_target',
            source_frame_policy='ae_large_preserved_free_graph_common_A_before_U')
    raise ValueError('Unknown paired field/target/frame policy')


def _sha256_file(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _target_entry(data, control):
    """Resolve an explicit target by its authenticated pseudopotential source.

    Paths are absolute preparation inputs, not search hints. A target does
    not replace the original PS table or its dual; it selects the offline
    physical-field constructor. The legacy name 'free_graph' denotes the
    normalized RKB graph, not exact free-Dirac spectral decoupling.
    """
    target = control.get("target")
    if target is None:
        return None
    if (not isinstance(target, dict) or set(target) != {"kind", "species"}
            or target["kind"] != AE_LARGE_TARGET or not isinstance(target["species"], dict)
            or not target["species"]):
        raise ValueError("unknown or incomplete normalized-cache target descriptor")
    for source, entry in target["species"].items():
        if (not isinstance(source, str) or len(source) != 64
                or any(c not in "0123456789abcdef" for c in source)
                or not isinstance(entry, dict) or set(entry) != _TARGET_KEYS
                or entry["source_upf_sha256"] != source):
            raise ValueError("target species requires a closed descriptor keyed by source UPF SHA256")
        for stem in ("matched_dirac", "pseudo_exterior", "source_upf"):
            path, sha = entry[stem + "_file"], entry[stem + "_sha256"]
            if (not isinstance(path, str) or not Path(path).is_absolute()
                    or not isinstance(sha, str) or len(sha) != 64
                    or any(c not in "0123456789abcdef" for c in sha)):
                raise ValueError("target primitive requires an absolute file and explicit SHA256")
        z = entry["nuclear_charge"]
        radii = [entry[key] for key in ("dirac_window_start", "dirac_window_stop",
                                       "completion_start", "completion_stop")]
        if (isinstance(z, bool) or not isinstance(z, int) or z <= 0
                or not np.isfinite(radii).all()
                or not 0 < radii[0] < radii[1] <= radii[2] < radii[3]):
            raise ValueError("target nuclear charge/window/completion controls are invalid")
    source = data["metadata"]["source_sha256"]
    if source not in target["species"]:
        raise ValueError("target descriptor does not cover the authenticated atomic source")
    return target["species"][source]


def _target_inputs(data, control, support_radius):
    """Authenticate the paired P/Q, exterior PS and unchanged original PCA."""
    from psp.atomic_reconstruction import load_atomic_reconstruction, upf_identity
    import xml.etree.ElementTree as ET

    entry = _target_entry(data, control)
    if entry is None:
        return None
    for stem in ("matched_dirac", "pseudo_exterior", "source_upf"):
        if _sha256_file(entry[stem + "_file"]) != entry[stem + "_sha256"]:
            raise ValueError("target primitive file SHA256 mismatch: " + stem)
    source = upf_identity(entry["source_upf_file"])
    if source["atomic_number"] != entry["nuclear_charge"]:
        raise ValueError("target nuclear charge differs from authenticated UPF")
    root = ET.parse(entry["source_upf_file"]).getroot()
    beta = root.find("PP_NONLOCAL")
    cutoffs = [float(node.attrib["cutoff_radius"]) for node in (() if beta is None else beta)
               if node.tag.startswith("PP_BETA") and "cutoff_radius" in node.attrib]
    if not cutoffs or entry["dirac_window_start"] < max(cutoffs):
        raise ValueError("target Dirac window would alter the declared UPF projector support")
    radial_node = root.find("PP_MESH/PP_R")
    if radial_node is None:
        raise ValueError("target UPF lacks the projector radial mesh")
    radial = np.fromstring((radial_node.text or "").replace("D", "E"), sep=" ")
    if not len(radial) or not np.isfinite(radial).all() or np.any(np.diff(radial) <= 0):
        raise ValueError("target UPF projector radial mesh is invalid")
    for node in beta:
        if node.tag.startswith("PP_BETA"):
            values = np.fromstring((node.text or "").replace("D", "E"), sep=" ")
            if not len(values) or len(values) > len(radial) or not np.isfinite(values).all():
                raise ValueError("target UPF projector radial payload is invalid")
            live = np.flatnonzero(values)
            if len(live) and entry["dirac_window_start"] < radial[live[-1]]:
                raise ValueError("target Dirac window would alter actual nonzero UPF projector support")
    paired = load_atomic_reconstruction(entry["matched_dirac_file"], entry["source_upf_file"])
    if any(key not in paired or not np.array_equal(value, paired[key])
           for key, value in data.items() if key != "metadata"):
        raise ValueError("target paired P/Q changed original native fields, amplitudes or PCA coefficients")
    qmeta = paired["metadata"].get("dev_matched_dirac_Q_OPF", {})
    if (qmeta.get("schema") != "lorrax.dev.matched_dirac_q_opf.v1"
            or qmeta.get("native_payload_sha256") != data["metadata"]["payload_sha256"]
            or qmeta.get("source_UPF_sha256") != entry["source_upf_sha256"]):
        raise ValueError("target matched P/Q lacks exact native-parent/source provenance")
    with np.load(entry["pseudo_exterior_file"], allow_pickle=False) as archive:
        exterior_meta = json.loads(str(archive["metadata_json"]))
        exterior = {key: archive[key].copy() for key in archive.files if key != "metadata_json"}
    if (exterior_meta.get("schema") != "lorrax.dev.native_ps_exterior.v1"
            or exterior_meta.get("source_UPF_sha256") != entry["source_upf_sha256"]
            or exterior_meta.get("native_payload_sha256") != data["metadata"]["payload_sha256"]
            or exterior_meta.get("bank_sha256") != qmeta.get("raw_bank_sha256")):
        raise ValueError("target PS exterior differs from the matched native source bank")
    n, columns = len(data["r"]), len(data["l"])
    er = exterior["r"]
    if (er.dtype != np.float64 or er.ndim != 1 or not np.isfinite(er).all()
            or np.any(er <= 0) or np.any(np.diff(er) <= 0)
            or not np.array_equal(er[:n], data["r"])
            or not np.array_equal(exterior["ell"], data["l"])
            or not np.array_equal(exterior["kappa"], data["kappa"])):
        raise ValueError("target exterior radial grid or angular labels differ")
    for key in ("ps_u", "ps_du_dr"):
        a = exterior[key]
        if (a.shape != (len(er), columns) or not np.isfinite(a).all()
                or not np.array_equal(a[:n], data[key])):
            raise ValueError("target exterior changed native PS amplitudes or derivatives")
    for key in ("ae_small_u", "ae_small_du_dr"):
        if paired[key].shape != data["ps_u"].shape or not np.isfinite(paired[key]).all():
            raise ValueError("target paired Q is not finite on the original OPF grid")
    qchannels, pchannels = qmeta.get("channels", []), exterior_meta.get("channels", [])
    if len(qchannels) != len(data["channel_nopf"]) or len(pchannels) != len(qchannels):
        raise ValueError("target source PCA channel inventory differs")
    for i, (l, k, rank) in enumerate(zip(data["channel_l"], data["channel_kappa"], data["channel_nopf"])):
        coeff_sha = hashlib.sha256(np.ascontiguousarray(data[f"channel_{i}_coefficients"]).tobytes()).hexdigest()
        q, p = qchannels[i], pchannels[i]
        if ((q.get("l"), q.get("kappa"), q.get("rank")) != (int(l), int(k), int(rank))
                or (p.get("ell"), p.get("kappa"), p.get("rank")) != (int(l), int(k), int(rank))
                or q.get("PCA_coefficients_sha256") != coeff_sha
                or p.get("coefficients_sha256") != coeff_sha):
            raise ValueError("target source PCA coefficient or channel binding differs")
        energies = paired[f"channel_{i}_training_energies_ps_ha"]
        if (not np.isfinite(energies).all()
                or energies.shape != (data[f"channel_{i}_coefficients"].shape[0],)
                or not np.array_equal(energies, exterior[f"channel_{i}_training_energies"])):
            raise ValueError("target paired/exterior training energies differ")
    native = float(data["r"][-1])
    if not (entry["dirac_window_stop"] == native <= entry["completion_start"]
            < entry["completion_stop"] <= er[-1]
            and native <= control["taper_start"] < support_radius < control["radius_max"]):
        raise ValueError("target native window, virtual completion and compact support differ")
    if entry["nuclear_charge"] / 137.036 >= np.min(np.abs(data["kappa"])):
        raise ValueError("target point-nuclear radial exponent is not real")
    import scipy
    binding = dict(kind=AE_LARGE_TARGET, descriptor=entry, scipy_version=scipy.__version__,
        matched_metadata_sha256=hashlib.sha256(_json_bytes(paired["metadata"])).hexdigest(),
        exterior_metadata_sha256=hashlib.sha256(_json_bytes(exterior_meta)).hexdigest(),
        graph="normalized_rkb_X=sigma.p/(2c); not exact free-Dirac spectral graph",
        normalization="matched ONCV large-only amplitudes; original PS dual/PCA unchanged",
        native_lower_role="diagnostic only; served lower is sigma-gradient of compact large")
    return paired, exterior, entry, binding


def _json_bytes(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def _controls(control, support_radius):
    result = {key: value for key, value in control.items() if key != "species_files"}
    km, nk = float(result["momentum_max"]), int(result["momentum_points"])
    rm, nr = float(result["radius_max"]), int(result["radius_points"])
    if "taper_start" not in result:
        raise ValueError("compact normalized cache requires an explicit taper_start preserving the native core")
    start = float(result["taper_start"])
    if (not np.isfinite((km, rm, support_radius)).all() or km <= 0 or nk < 4
            or rm <= support_radius or nr < 8 or support_radius <= 0
            or nk != result["momentum_points"] or nr != result["radius_points"]
            or not np.isfinite(start) or not 0 < start < support_radius):
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
    from psp.augmentation_spinors import COMPACT_GRAPH_FIELD_MODEL

    metadata = data["metadata"]
    if (metadata.get("operator_comparison", {}).get("authenticated") is not True
            or metadata.get("phase_branch_validated") is not True):
        raise ValueError("normalized cache requires authenticated atomic source and phase branch")
    owners = {}
    for module in ("common.bispinor_init", "common.gamma_matrices", "psp.augmentation_spinors",
                   "psp.atomic_reconstruction", "psp.augmentation_cache"):
        origin = importlib.util.find_spec(module).origin
        owners[module] = hashlib.sha256(Path(origin).read_bytes()).hexdigest()
    controls = _controls(control, support_radius)
    if float(data["r"][-1]) > float(controls["taper_start"]):
        raise ValueError("compact taper would alter the authenticated native reconstruction sphere")
    binding = {"schema": SCHEMA, "carrier": "normalized_rkb",
            "carrier_provenance": NORMALIZED_RKB_LIFT_PROVENANCE,
            "field_model": COMPACT_GRAPH_FIELD_MODEL,
            "pauli_precursor": "R^-1(w R delta_phi)",
            "tail_diagnostic_field": "unwindowed_hankel",
            "half_alpha_fs": float(HALFALPHA), "owner_sources_sha256": owners,
            "atomic_source_sha256": metadata["source_sha256"],
            "atomic_payload_sha256": metadata["payload_sha256"],
            "atomic_metadata_sha256": hashlib.sha256(_json_bytes(metadata)).hexdigest(),
            "controls": controls, "support_radius": float(support_radius),
            "units": {"radius": "bohr", "momentum": "bohr^-1",
                      "radial_wavefunction": "bohr^-3/2", "radial_derivative": "bohr^-5/2"}}
    inputs = _target_inputs(data, controls, support_radius)
    binding["target_kind"] = NATIVE_PAULI_TARGET if inputs is None else AE_LARGE_TARGET
    if inputs is not None:
        binding["target"] = inputs[3]
    return binding


def _validate_arrays(cache, data, control, *, support_radius):
    from common.bispinor_init import HALFALPHA
    from psp.augmentation_spinors import COMPACT_GRAPH_FIELD_MODEL
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
    expected = dict(field_model=COMPACT_GRAPH_FIELD_MODEL, taper_start=float(control['taper_start']),
                    support_radius=float(support_radius), half_alpha=float(HALFALPHA))
    if any(np.asarray(cache[name]).shape != () or cache[name] != value for name, value in expected.items()):
        raise ValueError("normalized cache compact field descriptor differs from controls")


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
    """Build the raw Hankel samples and explicit compact graph exactly once."""
    from scipy.special import roots_legendre
    from psp.atomic_reconstruction import evaluate_radial_correction
    from psp.augmentation_spinors import build_normalized_radial_cache, COMPACT_GRAPH_FIELD_MODEL
    from common.bispinor_init import HALFALPHA

    control = _controls(control, support_radius)
    if float(data["r"][-1]) > float(control["taper_start"]):
        raise ValueError("atomic pre-lift reconstruction radius exceeds compact taper start")
    inputs = _target_inputs(data, control, support_radius)
    if inputs is not None:
        cache = _build_ae_large_cache(*inputs[:3], control, support_radius)
        _validate_arrays(cache, data, control, support_radius=support_radius)
        return cache
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
    cache.update(field_model=np.asarray(COMPACT_GRAPH_FIELD_MODEL),
        taper_start=np.asarray(float(control['taper_start'])),
        support_radius=np.asarray(float(support_radius)), half_alpha=np.asarray(float(HALFALPHA)))
    _validate_arrays(cache, data, control, support_radius=support_radius)
    return cache


def _panels(radius, stop, breaks, order, momentum_max):
    edges=np.asarray(radius)[np.asarray(radius)<stop]
    for boundary in (stop,*breaks):
        edges=edges[abs(edges-boundary)>8*np.spacing(abs(boundary))]
    edges=np.unique(np.r_[0.,edges,stop,breaks])
    resolved=[edges[0]]
    for left,right in zip(edges[:-1],edges[1:]):
        n=max(1,int(np.ceil(momentum_max*(right-left)/8)))
        resolved.extend(np.linspace(left,right,n+1)[1:])
    edges=np.asarray(resolved)
    x,w=np.polynomial.legendre.leggauss(order)
    half,mid=np.diff(edges)/2,(edges[1:]+edges[:-1])/2
    return (mid[:,None]+half[:,None]*x).ravel(),(half[:,None]*w).ravel()


def _radial_u(radius, u, du, query, exponent):
    from scipy.interpolate import CubicHermiteSpline
    r=np.asarray(radius);inside=query<r[0]
    value=np.empty((len(query),u.shape[1]),complex)
    value[~inside]=CubicHermiteSpline(r,u,du,axis=0,extrapolate=False)(query[~inside])
    value[inside]=u[0][None]*(query[inside,None]/r[0])**exponent[None]
    if not np.isfinite(value).all():
        raise ValueError('target radial queries exceed the declared source continuation')
    return value/query[:,None]


def _radial_R_derivative(radius, u, du, query, exponent):
    """Differentiate the SAME u-Hermite/nuclear continuation as _radial_u."""
    from scipy.interpolate import CubicHermiteSpline
    r = np.asarray(radius)
    inside = query < r[0]
    value = np.empty((len(query), u.shape[1]), complex)
    derivative = np.empty_like(value)
    spline = CubicHermiteSpline(r,u,du,axis=0,extrapolate=False)
    value[~inside] = spline(query[~inside])
    derivative[~inside] = spline(query[~inside],1)
    value[inside] = u[0][None]*(query[inside,None]/r[0])**exponent[None]
    derivative[inside] = value[inside]*exponent[None]/query[inside,None]
    result = derivative/query[:,None]-value/query[:,None]**2
    if not np.isfinite(result).all():
        raise ValueError('target derivative queries exceed the declared source continuation')
    return result


def _momentum(control):
    from scipy.special import roots_legendre
    maximum,count=float(control['momentum_max']),int(control['momentum_points'])
    if maximum<=0 or count<8 or count!=control['momentum_points']:
        raise ValueError('invalid projected-target momentum controls')
    if control.get('momentum_quadrature')=='gauss_legendre':
        nodes,weights=roots_legendre(count)
        return maximum*(nodes+1)/2,maximum*weights/2
    if control.get('momentum_quadrature','midpoint')!='midpoint':
        raise ValueError('unknown projected-target momentum quadrature')
    return (np.arange(count)+.5)*maximum/count,np.full(count,maximum/count)


def _inverse_pauli_spectrum(spectrum, ell, kappa, K, weights, radius):
    from scipy.special import spherical_jn
    from psp.augmentation_spinors import _lift_cartesian
    pauli=np.zeros((len(ell),2,len(K)),complex);pauli[:,0]=spectrum.T
    four=_lift_cartesian(pauli,np.column_stack((0*K,0*K,K)))
    upper_spectrum,lower_spectrum=four[:,0].T,four[:,2].T
    cache=dict(radius=radius,ell=ell.copy(),kappa=kappa.copy())
    for name in ('large_R','dlarge_R_dr','small_R','dsmall_R_dr'):
        cache[name]=np.empty((len(radius),len(ell)),complex)
    kr=radius[:,None]*K
    weight=(2/np.pi)*weights*K*K
    for k in np.unique(kappa):
        ids=np.flatnonzero(kappa==k);l=int(ell[ids[0]]);lb=2*abs(k)-1-l
        a=weight[:,None]*upper_spectrum[:,ids]
        b=-1j**(lb-l)*weight[:,None]*lower_spectrum[:,ids]
        cache['large_R'][:,ids]=spherical_jn(l,kr)@a
        cache['dlarge_R_dr'][:,ids]=spherical_jn(l,kr,derivative=True)@(K[:,None]*a)
        cache['small_R'][:,ids]=spherical_jn(lb,kr)@b
        cache['dsmall_R_dr'][:,ids]=spherical_jn(lb,kr,derivative=True)@(K[:,None]*b)
    return cache


def _ae_large_spectral_target(data, pseudo_exterior, entry, control, support_radius):
    """Return the existing AE-large construction and its single Pauli precursor.

    Implicit Pauli correction is R_inverse DeltaL. Canonical U is applied
    exactly once and its R cancels R_inverse; no pointwise normalization.
    Native Dirac Q is a diagnostic, not an inserted kinetic-balance lower.

    pseudo_exterior supplies r,ps_u,ps_du_dr,ell,kappa in the exact original
    OPF order and amplitude. Its interior rows must be byte-identical. The
    input bank's ONCV large-only matching convention is retained throughout.
    """
    from psp.augmentation_spinors import (build_normalized_radial_cache,
        compact_graph_taper,radial_fourier_bessel,_lift_cartesian,free_graph_small_from_large)
    nuclear_charge = entry['nuclear_charge']
    dirac_window_start, dirac_window_stop = entry['dirac_window_start'], entry['dirac_window_stop']
    completion_start, completion_stop = entry['completion_start'], entry['completion_stop']
    r=np.asarray(data['r']);l=np.asarray(data['l']);k=np.asarray(data['kappa'])
    er=np.asarray(pseudo_exterior['r'])
    if (not np.array_equal(er[:len(r)],r)
            or not np.array_equal(pseudo_exterior['ps_u'][:len(r)],data['ps_u'])
            or not np.array_equal(pseudo_exterior['ps_du_dr'][:len(r)],data['ps_du_dr'])
            or not np.array_equal(pseudo_exterior['ell'],l)
            or not np.array_equal(pseudo_exterior['kappa'],k)):
        raise ValueError('pseudo completion changed the native interior, amplitude or OPF order')
    if not (0<dirac_window_start<dirac_window_stop<=r[-1]
            <=completion_start<completion_stop<=er[-1]
            and r[-1]<=control['taper_start']<support_radius<=control['radius_max']):
        raise ValueError('Dirac window, virtual completion and physical compact support are inconsistent')
    K,wK=_momentum(control);order=int(control.get('source_quadrature_order',16))
    if order<12:
        raise ValueError('projected target requires resolved source quadrature')
    qr,qw=_panels(er,completion_stop,(completion_start,),order,float(K[-1]))
    pseudo=_radial_u(er,pseudo_exterior['ps_u'],pseudo_exterior['ps_du_dr'],qr,l+1)
    virtual=compact_graph_taper(qr,completion_start,completion_stop)[0]
    pseudo*=virtual[:,None]
    dr,dw=_panels(r,dirac_window_stop,(dirac_window_start,),order,float(K[-1]))
    # The declared virtual pseudo reference is unfolded once with canonical U.
    ps4=build_normalized_radial_cache(pseudo,qr,qw,l,k,K,wK,dr)
    gamma=np.sqrt(k*k-(float(nuclear_charge)/137.036)**2)
    AE=_radial_u(r,data['ae_u'],data['ae_du_dr'],dr,gamma)
    dAE=_radial_R_derivative(r,data['ae_u'],data['ae_du_dr'],dr,gamma)
    Q=_radial_u(r,data['ae_small_u'],data['ae_small_du_dr'],dr,gamma)
    window,dwindow=compact_graph_taper(dr,dirac_window_start,dirac_window_stop)
    raw_large=AE-ps4['large_R'];raw_small=1j*Q-ps4['small_R']
    delta_large=window[:,None]*raw_large
    ddelta_large=dwindow[:,None]*raw_large+window[:,None]*(dAE-ps4['dlarge_R_dr'])
    lower_free_graph=free_graph_small_from_large(dr,delta_large,ddelta_large,k)
    delta_small=window[:,None]*raw_small
    AL=radial_fourier_bessel(delta_large,dr,dw,l,K)
    AQ=radial_fourier_bessel(-1j*delta_small,dr,dw,2*abs(k)-1-l,K)
    unit=np.zeros((1,2,len(K)),complex);unit[0,0]=1.
    R=_lift_cartesian(unit,np.column_stack((0*K,0*K,K)))[0,0].real
    # Preserve large AL: canonical U(R_inverse AL) has large AL and
    # lower sigma.p AL/(2c). AQ remains a native-Dirac diagnostic only.
    pauli=AL/R[:,None]
    return dict(momentum=K, weights_dK=wK, pauli_radial_spectrum=pauli,
        target_large_spectrum=AL, native_small_diagnostic_spectrum=AQ,
        source_radius=dr, source_weights_dr=dw, source_large_R=delta_large,
        source_dlarge_R_dr=ddelta_large, source_lower_free_graph_R=lower_free_graph,
        source_native_small_R=delta_small, ell=l.copy(), kappa=k.copy())


def _build_ae_large_cache(data, pseudo_exterior, entry, control, support_radius):
    """Preserve the existing AE-large cache via its shared spectral target owner."""
    from common.bispinor_init import HALFALPHA
    from psp.augmentation_spinors import COMPACT_GRAPH_FIELD_MODEL
    target = _ae_large_spectral_target(data,pseudo_exterior,entry,control,support_radius)
    cache=_inverse_pauli_spectrum(target['pauli_radial_spectrum'],target['ell'],target['kappa'],
        target['momentum'],target['weights_dK'],_radius_grid(control))
    cache.update(field_model=np.asarray(COMPACT_GRAPH_FIELD_MODEL),
        taper_start=np.asarray(float(control['taper_start'])),support_radius=np.asarray(float(support_radius)),
        half_alpha=np.asarray(float(HALFALPHA)))
    return cache


def build_ae_large_paired_cache(data, control, *, support_radius):
    """Prepare unwindowed chi and U chi from the authenticated AE-large target.

    The construction window remains owned by ``_ae_large_spectral_target``.
    Neither paired inverse is tapered after U. The returned explicit target
    binding and spectral witnesses distinguish construction from a caller's
    finite serving domain; this utility makes no sphere/tail accuracy claim.
    Native Dirac small waves remain diagnostics, never a second lower field.
    """
    from psp.augmentation_spinors import build_paired_radial_cache_from_spectrum

    controls = _controls(control, support_radius)
    inputs = _target_inputs(data, controls, support_radius)
    if inputs is None:
        raise ValueError('AE-large paired preparation requires its explicit authenticated target')
    gamma = np.sqrt(np.asarray(data['kappa'])**2-(inputs[2]['nuclear_charge']/137.036)**2)
    if np.any(gamma <= .5):
        raise ValueError('Point-nuclear AE-large Sobolev target requires gamma > 1/2; finite-nucleus data are not implied')
    target = _ae_large_spectral_target(*inputs[:3], controls, support_radius)
    pair = build_paired_radial_cache_from_spectrum(target['pauli_radial_spectrum'],
        target['ell'], target['kappa'], target['momentum'], target['weights_dK'],
        _radius_grid(controls))
    return dict(pair, target_binding=inputs[3], construction_controls=controls,
        target_large_spectrum=target['target_large_spectrum'],
        native_small_diagnostic_spectrum=target['native_small_diagnostic_spectrum'],
        source_radius=target['source_radius'], source_weights_dr=target['source_weights_dr'],
        source_large_R=target['source_large_R'],
        source_dlarge_R_dr=target['source_dlarge_R_dr'],
        source_lower_free_graph_R=target['source_lower_free_graph_R'],
        source_native_small_R=target['source_native_small_R'])


def normalized_cache_tail_diagnostics(cache, *, support_radius):
    """Unwindowed Hankel tail estimates; no compact-field accuracy claim."""
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
    _validate_arrays(cache, data, binding["controls"], support_radius=support_radius)
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
    _validate_arrays(cache, data, expected["controls"], support_radius=support_radius)
    if metadata.get("payload_sha256") != _payload_hash(cache):
        raise ValueError("normalized cache array payload checksum mismatch")
    return cache


def load_paired_native_cache(path, data, *, expected_file_sha256,
                             common_spectrum_sha256, carrier, support_radius):
    """Admit raw paired Hermite fields as approximations to one compact chi.

    This explicit field policy differs from the legacy compact-after-U
    graph. It neither windows the fields after U nor substitutes their
    spectral or finite-sphere Gram for the compact target Gram. The raw
    evaluator retains independent large/small derivatives and tail
    diagnostics. No field rescaling occurs here.
    """
    return load_paired_field_cache(path, data,
        expected_file_sha256=expected_file_sha256,
        common_spectrum_sha256=common_spectrum_sha256, carrier=carrier,
        support_radius=support_radius, field_policy=PAIRED_COMPACT_PAULI_FIELD_POLICY)


def load_paired_field_cache(path, data, *, expected_file_sha256,
                            common_spectrum_sha256, carrier, support_radius,
                            field_policy):
    """Authenticate raw paired fields under their explicit closed target.

    The AE-large schema cannot reuse the native target certificate. Its
    primitive target binding is re-read by the same constructor input owner;
    no post-U serving window or normalization is added by this loader.
    """
    import inspect
    from psp.augmentation_spinors import evaluate_normalized_radials
    from common.bispinor_init import lift_to_4spinor
    from common import gamma_matrices

    contract = paired_field_policy_contract(field_policy)

    if carrier not in ('pauli2embed4', 'normalized_rkb'):
        raise ValueError('paired native cache requires an explicit Pauli or canonical-U carrier')
    if (not np.isfinite(support_radius) or support_radius <= 0
            or any(not isinstance(sha,str) or len(sha)!=64
                or any(c not in '0123456789abcdef' for c in sha)
                for sha in (expected_file_sha256,common_spectrum_sha256))):
        raise ValueError('paired native cache requires a positive support and explicit SHA256 bindings')
    if _sha256_file(path) != expected_file_sha256:
        raise ValueError('paired native field file identity mismatch')
    with np.load(path, allow_pickle=False) as archive:
        metadata = json.loads(str(archive['metadata_json']))
        cache = {key: archive[key].copy() for key in archive.files if key != 'metadata_json'}
    raw_keys = frozenset(('radius', 'ell', 'kappa', *RADIAL_KEYS))
    if (set(cache) != raw_keys
            or metadata.get('schema') != contract['field_schema']
            or metadata.get('source_model') != contract['field_source_model']
            or metadata.get('no_post_U_taper') is not True
            or metadata.get('carrier') != carrier
            or metadata.get('source_upf_sha256') != data['metadata']['source_sha256']
            or metadata.get('native_payload_sha256') != data['metadata']['payload_sha256']
            or metadata.get('common_spectrum_sha256') != common_spectrum_sha256
            or metadata.get('payload_sha256') != _payload_hash(cache)):
        raise ValueError('paired native field policy/source/spectrum/payload mismatch')
    controls = metadata.get('controls', {})
    if field_policy == PAIRED_AE_LARGE_FIELD_POLICY:
        construction = metadata.get('construction_controls')
        target = metadata.get('target_binding')
        if (metadata.get('target_kind') != AE_LARGE_TARGET
                or not isinstance(construction, dict) or not isinstance(target, dict)):
            raise ValueError('AE-large paired fields require the closed constructor controls and target binding')
        inputs = _target_inputs(data, construction, support_radius)
        if inputs is None or inputs[3] != target:
            raise ValueError('AE-large paired field primitive target binding mismatch')
        gamma = np.sqrt(np.asarray(data['kappa'])**2-(inputs[2]['nuclear_charge']/137.036)**2)
        if np.any(gamma <= .5):
            raise ValueError('Point-nuclear AE-large Sobolev target requires gamma > 1/2')
        witness, witness_sha = metadata.get('target_witness_file'), metadata.get('target_witness_sha256')
        if (not isinstance(witness, str) or not Path(witness).is_absolute()
                or not isinstance(witness_sha, str) or len(witness_sha) != 64
                or any(c not in '0123456789abcdef' for c in witness_sha)
                or _sha256_file(witness) != witness_sha):
            raise ValueError('AE-large paired fields require their pinned construction-target witness')
    r = np.asarray(cache['radius'])
    if (controls.get('support_radius') != support_radius or r.dtype != np.float64
            or r.ndim != 1 or len(r) < 2 or not np.isfinite(r).all()
            or r[0] != 0 or np.any(np.diff(r) <= 0) or r[-1] <= support_radius
            or not np.array_equal(cache['ell'], data['l'])
            or not np.array_equal(cache['kappa'], data['kappa'])):
        raise ValueError('paired native field radius/support/native channel inventory mismatch')
    for key in RADIAL_KEYS:
        value = cache[key]
        if (value.dtype != np.complex128 or value.shape != (len(r), len(data['l']))
                or not np.isfinite(value).all()):
            raise ValueError('paired native Hermite values/derivatives must be finite complex128')
    if carrier == 'pauli2embed4' and (np.any(cache['small_R']) or np.any(cache['dsmall_R_dr'])):
        raise ValueError('paired Pauli embedding must have exactly zero lower field and derivative')
    owners = metadata.get('source_owners_sha256')
    if not isinstance(owners, dict) or not owners:
        raise ValueError('paired native cache requires declared physical field/lift source owners')
    for owner in (evaluate_normalized_radials, lift_to_4spinor, gamma_matrices):
        name = inspect.getsourcefile(owner)
        actual = _sha256_file(name)
        recorded = {sha for source, sha in owners.items() if Path(source).name == Path(name).name}
        if recorded != {actual}:
            raise ValueError('paired native field or canonical-U source owner changed')
    binding = dict(policy=field_policy, carrier=carrier,
        file_sha256=expected_file_sha256, common_spectrum_sha256=common_spectrum_sha256,
        native_reconstruction_sha256=metadata['native_reconstruction_sha256'],
        native_payload_sha256=metadata['native_payload_sha256'], source_upf_sha256=metadata['source_upf_sha256'],
        controls=controls, no_post_U_taper=True, represented_domain_bohr=[float(r[0]), float(r[-1])],
        tail_diagnostics=normalized_cache_tail_diagnostics(cache, support_radius=support_radius))
    if field_policy == PAIRED_AE_LARGE_FIELD_POLICY:
        binding.update(target_kind=AE_LARGE_TARGET, target_binding=target,
            construction_controls=construction, field_schema=contract['field_schema'],
            target_witness_file=witness, target_witness_sha256=witness_sha)
    return cache, binding
