"""Authenticated atomic partial waves for norm-conserving reconstruction.

An ONCV Kleinman--Bylander projector is an operator factor, not a dual
wavefunction reconstruction projector.  This module instead orthonormalizes
the *pseudo partial waves* and applies exactly the same transformation to
their paired all-electron waves, following the OCEAN optimal-projector
construction.  Arrays are small host tables generated once per species.

Radial wavefunctions use ``u(r) = r R(r)``; derivatives are physical
``du/dr``, never derivatives with respect to an atomic log-mesh index.  AE
waves describe the ONCV large-component reference.  The actual normalized
four-component lift belongs to the relativistic carrier, not this table.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re
import xml.etree.ElementTree as ET

import numpy as np


SCHEMA = "lorrax.atomic_reconstruction.v1"


def generated_upf_text(oncv_log: str | Path) -> str:
    """Extract one completed UPF member from a successful ONCV receipt.

    Reference-input files with zero additional test configurations are
    accepted; a missing output member or any atomic/ghost failure is refused.
    The extraction itself establishes no transferability certificate.
    """
    text = Path(oncv_log).read_text()
    failures = [line for line in text.splitlines()
                if re.search(r"\bERROR(?:[:\s,])|WARNING\s*-\s*GHOST\([+-]\)|Fortran runtime error", line)]
    if failures:
        raise ValueError("ONCV generation failed: " + "; ".join(failures[:8]))
    if text.count("PSP_UPF") != 1:
        raise ValueError("ONCV receipt must contain exactly one UPF member")
    member = text.split("PSP_UPF", 1)[1]
    if "END_PSP" not in member:
        raise ValueError("ONCV receipt contains an incomplete UPF member")
    member = member.split("END_PSP", 1)[0].strip() + "\n"
    ET.fromstring(member)
    return member


def embedded_oncv_input(upf_path: str | Path) -> str:
    """Return the complete atomic generator input embedded in a UPF file."""
    root = ET.parse(upf_path).getroot()
    node = root.find("PP_INFO/PP_INPUTFILE")
    if node is None or not (node.text or "").strip():
        raise ValueError("atomic reconstruction requires PP_INPUTFILE")
    return node.text.strip() + "\n"


def restored_oncv_input(upf_path: str | Path, reference_log: str | Path) -> tuple[str, dict]:
    """Undo ONCV's j-specific ``debl`` adjustment before replaying its echo.

    The embedded input is printed *after* the reference atom sets
    ``debl_j = debl_input + Ebar - E_j``.  Feeding that echo back applies the
    shift twice.  The independent reference-atom energies recover the actual
    input on ONCV's printed five-decimal grid.  Bound second projectors are
    subsequently set to their atomic eigenenergy by ONCV itself.
    """
    rows = [line.split() for line in embedded_oncv_input(upf_path).splitlines()
            if line.strip() and not line.lstrip().startswith("#")]
    nc, nv = map(int, rows[0][2:4])
    log = Path(reference_log).read_text().split("# PSEUDOPOTENTIAL AND OPTIMIZATION", 1)[0]
    energies = {}
    for line in log.splitlines():
        words = line.split()
        if len(words) in (4, 5) and words[0].isdigit() and words[1].isdigit() and "D" in words[3]:
            n, l = map(int, words[:2])
            energies[n, l] = [float(word.replace("D", "E")) for word in words[3:]]
    first_valence = {}
    for row in rows[1 + nc:1 + nc + nv]:
        n, l = map(int, row[:2])
        if (n, l) not in energies:
            raise ValueError("ONCV reference receipt lacks a required atomic eigenenergy")
        first_valence.setdefault(l, energies[n, l])
    lmax = int(rows[1 + nc + nv][0])
    projector_start = 1 + nc + nv + 1 + lmax + 1 + 1
    changes = []
    for index in range(projector_start, projector_start + lmax + 1):
        l = int(rows[index][0])
        if l > 0 and l in first_valence:
            e = first_valence[l]
            if len(e) != 2:
                raise ValueError("ONCV reconstruction requires both j-resolved reference energies")
            mean = ((l + 1) * e[0] + l * e[1]) / (2 * l + 1)
            printed = float(rows[index][2])
            original = round(printed + e[0] - mean, 5)
            rows[index][2] = f"{original:.5f}"
            changes.append({"l": l, "echo_debl": printed, "input_debl": original,
                            "j_plus_energy_ha": e[0], "j_minus_energy_ha": e[1]})
    return ("\n".join(" ".join(row) for row in rows) + "\n",
            {"reason": "invert ONCV post-generation j-specific debl echo", "changes": changes})


def read_atomic_scattering_bank(path: str | Path) -> dict:
    """Read the generation tool's completed, kappa-resolved radial bank.

    Returned ``*_u`` and ``*_du_dr`` arrays have shape
    ``(n_channel, n_energy, n_r)``.  Training rows and independently sampled
    held-out energies are explicitly distinguished by ``training_mask``.
    """
    with Path(path).open() as handle:
        if not handle.readline().startswith("# lorrax.atomic_scattering_bank.v1"):
            raise ValueError("unsupported atomic scattering bank")
        header = handle.readline().split()
        nch, ne, nr = map(int, header[:3])
        radius, charge = map(float, header[3:])
        descriptors, records, values = [], [], []
        for channel in range(nch):
            descriptor = handle.readline().split()
            if len(descriptor) != 6 or int(descriptor[0]) != channel + 1:
                raise ValueError("atomic scattering bank has an incomplete channel")
            descriptors.append([float(x) for x in descriptor[1:]])
            energy_rows, radial_rows = [], []
            for energy in range(ne):
                row = handle.readline().split()
                if len(row) != 8 or int(row[0]) != energy + 1:
                    raise ValueError("atomic scattering bank has an incomplete energy row")
                energy_rows.append([float(x) for x in row[1:]])
                radial = np.loadtxt(handle, max_rows=nr)
                if radial.shape != (nr, 7):
                    raise ValueError("atomic scattering bank has an incomplete radial row")
                radial_rows.append(radial)
            records.append(energy_rows)
            values.append(radial_rows)
        if handle.read().strip():
            raise ValueError("unexpected trailing atomic scattering data")
    values, records = np.asarray(values), np.asarray(records)
    descriptors = np.asarray(descriptors)
    if not np.all(np.isfinite(values)) or not np.all(np.isfinite(records)):
        raise ValueError("non-finite atomic scattering bank")
    r = values[0, 0, :, 0]
    if not np.all(values[..., 0] == r) or abs(r[-1] - radius) > 1e-12:
        raise ValueError("atomic scattering bank changed its radial quadrature")
    result = {"r": r, "weights_dr": log_radial_weights(r),
              "radius_augmentation": radius, "atomic_number": charge,
              "channel_l": descriptors[:, 0].astype(int),
              "channel_kappa": descriptors[:, 1].astype(int),
              "channel_nprojector": descriptors[:, 2].astype(int),
              "training_mask": records[:, :, 0].astype(bool),
              "node_reference": records[:, :, 1].astype(int),
              "node_matched": records[:, :, 2].astype(int),
              "energies_ps": records[:, :, 3], "energies_ae": records[:, :, 4],
              "boundary_phase_residual": records[:, :, 5]}
    for index, name in enumerate(("ps_u", "ps_du_dr", "ae_u", "ae_du_dr",
                                  "ae_small_u", "ae_small_du_dr"), start=1):
        result[name] = values[..., index]
    return result


def log_radial_weights(r: np.ndarray) -> np.ndarray:
    """Composite Simpson/3/8 ``dr`` weights on a uniform log-radius mesh.

    The last three intervals use the 3/8 rule when their total count is odd;
    no duplicated endpoint or trapezoidal last panel changes the convention.
    The excluded interval ``[0,r[0]]`` is below the generator's cusp cutoff.
    """
    r = np.asarray(r, dtype=float)
    if (r.ndim != 1 or len(r) < 4 or not np.all(np.isfinite(r))
            or np.any(r <= 0) or np.any(np.diff(r) <= 0)):
        raise ValueError("atomic radial quadrature requires at least four increasing positive radii")
    dx = np.diff(np.log(r))
    if np.max(np.abs(dx - dx[0])) > 1e-12:
        raise ValueError("atomic radial quadrature requires a uniform logarithmic mesh")
    n = len(r)
    weights = np.zeros(n)
    last = n - 1 if n % 2 else n - 4
    if last:
        weights[:last + 1] = 2
        weights[1:last:2] = 4
        weights[[0, last]] = 1
        weights[:last + 1] *= dx[0] / 3
    if n % 2 == 0:
        weights[-4:] += 3 * dx[0] / 8 * np.asarray((1, 3, 3, 1))
    return weights * r


def read_atomic_frozen_core(path: str | Path, r: np.ndarray) -> dict:
    """Read true frozen-core Dirac waves on the reconstruction sphere.

    ``core_u`` and ``core_small_u`` have shape ``(n_r,n_core_channel)``;
    occupations are apportioned between j channels by their degeneracy.
    ``core_dirac_norm_inside`` measures truncation against the bound solver's
    whole-space Dirac normalization, including both large and small waves.
    """
    labels, waves = [], []
    with Path(path).open() as handle:
        if not handle.readline().startswith("# lorrax.atomic_frozen_core.v1"):
            raise ValueError("unsupported atomic frozen-core table")
        nshell, nr = map(int, handle.readline().split())
        if nr != len(r):
            raise ValueError("atomic frozen-core radial mesh differs from the partial waves")
        if nshell == 0:
            if handle.read().strip():
                raise ValueError("zero-core atomic table contains unexpected orbitals")
            return {"core_n": np.empty(0, dtype=int), "core_l": np.empty(0, dtype=int),
                    "core_kappa": np.empty(0, dtype=int), "core_energy_ha": np.empty(0),
                    "core_occupation": np.empty(0), "core_u": np.empty((nr, 0)),
                    "core_small_u": np.empty((nr, 0)), "core_dirac_norm_inside": np.empty(0)}
        while True:
            row = handle.readline().split()
            if not row:
                break
            if len(row) != 6:
                raise ValueError("incomplete atomic frozen-core channel")
            labels.append([float(x) for x in row])
            values = np.loadtxt(handle, max_rows=nr)
            if values.shape != (nr, 3) or not np.array_equal(values[:, 0], r):
                raise ValueError("incomplete atomic frozen-core radial samples")
            waves.append(values[:, 1:])
        if handle.read().strip():
            raise ValueError("unexpected atomic frozen-core trailing data")
    labels, waves = np.asarray(labels), np.asarray(waves)
    if len(np.unique(labels[:, 0])) != nshell or not np.all(np.isfinite(waves)):
        raise ValueError("incomplete or non-finite atomic frozen-core table")
    l, kappa = labels[:, 2].astype(int), labels[:, 3].astype(int)
    occupation = labels[:, 5] * np.abs(kappa) / (2 * l + 1)
    w = log_radial_weights(r)
    return {"core_n": labels[:, 1].astype(int), "core_l": l,
            "core_kappa": kappa, "core_energy_ha": labels[:, 4],
            "core_occupation": occupation, "core_u": waves[:, :, 0].T,
            "core_small_u": waves[:, :, 1].T,
            "core_dirac_norm_inside": np.sum(w[None, :] * np.sum(waves ** 2, axis=2), axis=1)}


def _numeric_sections(root):
    sections = {}
    for node in root.iter():
        tag = node.tag
        if tag in {"PP_R", "PP_RAB", "PP_LOCAL", "PP_NLCC", "PP_DIJ", "PP_RHOATOM"} or tag.startswith(("PP_BETA.", "PP_CHI.")):
            values = np.fromstring((node.text or "").replace("D", "E"), sep=" ")
            if not len(values) or not np.all(np.isfinite(values)):
                raise ValueError(f"invalid numerical UPF section {tag}")
            sections[tag] = values
    return sections


def upf_identity(upf_path: str | Path) -> dict:
    """Fingerprint the source file, its atomic input and j-resolved operator."""
    path = Path(upf_path)
    root = ET.parse(path).getroot()
    header = root.find("PP_HEADER").attrib
    required = ("element", "pseudo_type", "relativistic", "has_so", "functional", "z_valence")
    identity = {key: header[key].strip() for key in required}
    input_rows = [line.split() for line in embedded_oncv_input(path).splitlines()
                  if line.strip() and not line.lstrip().startswith("#")]
    nc, nv = map(int, input_rows[0][2:4])
    configuration = [{"n": int(row[0]), "l": int(row[1]), "occupation": float(row[2])}
                     for row in input_rows[1:1 + nc + nv]]
    frozen = {"atomic_number": float(input_rows[0][1]), "functional": identity["functional"],
              "core_shells": configuration[:nc]}
    sections = _numeric_sections(root)
    digest = hashlib.sha256()
    for name, values in sorted(sections.items()):
        digest.update(name.encode())
        digest.update(np.asarray(values, dtype="<f8").tobytes())
    spin = root.find("PP_SPIN_ORB")
    channels = [dict(node.attrib) for node in spin] if spin is not None else []
    digest.update(json.dumps(channels, sort_keys=True).encode())
    identity.update(source_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                    operator_sha256=digest.hexdigest(), spin_channels=channels,
                    generator_input_sha256=hashlib.sha256(embedded_oncv_input(path).encode()).hexdigest())
    identity.update(atomic_number=float(input_rows[0][1]),
                    reference_configuration=configuration, number_core_shells=nc,
                    frozen_configuration_sha256=hashlib.sha256(json.dumps(frozen, sort_keys=True).encode()).hexdigest())
    return identity


def compare_regenerated_upf(source_path: str | Path, regenerated_path: str | Path,
                            *, relative_tolerance: float, absolute_tolerance: float) -> dict:
    """Authenticate regeneration against every source radial/operator table.

    Tolerances are explicit properties of a generation receipt.  A matching
    element or generator version alone never authenticates atomic data.
    ``PP_CHI`` and NLCC are compared too, although neither supplies AE waves.
    """
    source = upf_identity(source_path)
    regenerated = upf_identity(regenerated_path)
    for key in ("element", "pseudo_type", "relativistic", "has_so", "functional", "z_valence"):
        if source[key] != regenerated[key]:
            raise ValueError(f"atomic regeneration metadata mismatch: {key}")
    # ONCV prints an input echo with different spacing/version comments.
    def configuration(path):
        return [line.split() for line in embedded_oncv_input(path).splitlines()
                if line.strip() and not line.lstrip().startswith("#")]
    if configuration(source_path) != configuration(regenerated_path):
        raise ValueError("atomic regeneration changed the atomic configuration or generator parameters")
    def labels(path):
        root = ET.parse(path).getroot()
        spin = root.find("PP_SPIN_ORB")
        return [(n.tag, tuple(sorted(n.attrib.items()))) for n in spin]
    if labels(source_path) != labels(regenerated_path):
        raise ValueError("atomic regeneration changed j-resolved channel labels")
    left = _numeric_sections(ET.parse(source_path).getroot())
    right = _numeric_sections(ET.parse(regenerated_path).getroot())
    if left.keys() != right.keys():
        raise ValueError("atomic regeneration changed available radial/operator sections")
    rows = {}
    for name in left:
        a, b = left[name], right[name]
        if a.shape != b.shape:
            raise ValueError(f"atomic regeneration changed {name} shape")
        error = np.abs(a - b)
        scale = absolute_tolerance + relative_tolerance * np.abs(a)
        ratio = np.max(error / scale)
        rows[name] = {"max_absolute_error": float(np.max(error)),
                      "max_tolerance_ratio": float(ratio)}
        if ratio > 1:
            raise ValueError(f"atomic regeneration mismatch in {name}: {rows[name]}")
    return {"source": source, "regenerated": regenerated,
            "relative_tolerance": relative_tolerance,
            "absolute_tolerance": absolute_tolerance, "sections": rows,
            "authenticated": True}


def radial_operator_difference(source_path: str | Path, regenerated_path: str | Path) -> dict:
    """Bound the j-resolved regenerated radial Hamiltonian change in Hartree.

    For each ``(l,j)`` channel this evaluates the spectral norm of
    ``sqrt(w) (B D B.T - B' D' B'.T) sqrt(w)`` on the UPF radial mesh.
    It is invariant to KB projector rotations/signs.  The local potential
    contributes its pointwise supremum.  ``PP_DIJ``/``PP_LOCAL`` use Ry;
    the returned bounds use Ha.  This isolates generation drift and is
    independent of the later orbital reconstruction error.
    """
    roots = [ET.parse(path).getroot() for path in (source_path, regenerated_path)]
    sections = [_numeric_sections(root) for root in roots]
    r = sections[0]["PP_R"]
    if not np.array_equal(r, sections[1]["PP_R"]):
        raise ValueError("radial operator comparison requires the identical source mesh")
    # The UPF grid is linear and its stored beta is r times radial beta.
    weights = np.empty_like(r)
    weights[0] = (r[1] - r[0]) / 2
    weights[-1] = (r[-1] - r[-2]) / 2
    weights[1:-1] = (r[2:] - r[:-2]) / 2
    labels = [node for node in roots[0].find("PP_SPIN_ORB") if node.tag.startswith("PP_RELBETA.")]
    grouped = {}
    for node in labels:
        grouped.setdefault((int(node.get("lll")), float(node.get("jjj"))), []).append(int(node.get("index")) - 1)
    betas, dij = [], []
    for root, tables in zip(roots, sections):
        number = int(root.find("PP_HEADER").get("number_of_proj"))
        betas.append(np.column_stack([tables[f"PP_BETA.{i + 1}"] for i in range(number)]))
        dij.append(tables["PP_DIJ"].reshape(number, number) / 2)
    channels = []
    for (l, j), indices in grouped.items():
        support = np.any((betas[0][:, indices] != 0) | (betas[1][:, indices] != 0), axis=1)
        b, bprime = [table[support][:, indices] * np.sqrt(weights[support, None]) for table in betas]
        d, dprime = [matrix[np.ix_(indices, indices)] for matrix in dij]
        difference = b @ d @ b.T - bprime @ dprime @ bprime.T
        bound = float(np.max(np.abs(np.linalg.eigvalsh(difference))))
        channels.append({"l": l, "j": j, "nonlocal_spectral_bound_ha": bound})
    local = float(np.max(np.abs(sections[0]["PP_LOCAL"] - sections[1]["PP_LOCAL"])) / 2)
    return {"local_supremum_bound_ha": local, "channels": channels,
            "hamiltonian_bound_ha": local + max(row["nonlocal_spectral_bound_ha"] for row in channels)}


def optimal_radial_projectors(r: np.ndarray, weights_dr: np.ndarray,
                             paired_bank: dict, *, discarded_weight: float) -> dict:
    r"""Compute paired OCEAN optimal radial functions with shared PCA weights.

    Parameters
    ----------
    r, weights_dr : (n_r,) ndarray
        Positive radii in bohr and positive ``dr`` quadrature weights.
    paired_bank : dict
        ``ps_u``, ``ps_du_dr``, ``ae_u``, ``ae_du_dr`` each have shape
        ``(n_energy, n_r)``.  The AE boundary values and slopes must match
        the pseudo values and slopes.  Each row spans the same sphere.
    discarded_weight : float
        Maximum fraction of the *pseudo* partial-wave Gram trace discarded.
        This is a reconstruction diagnostic, not a self-energy certificate.

    Returns
    -------
    dict
        Paired ``*_u`` and physical radial derivatives with shape
        ``(n_r, n_opf)``, shared ``coefficients`` of shape
        ``(n_energy, n_opf)``, and independent overlap diagnostics.
    """
    r = np.asarray(r, dtype=np.float64)
    w = np.asarray(weights_dr, dtype=np.float64)
    names = ("ps_u", "ps_du_dr", "ae_u", "ae_du_dr")
    bank = {name: np.asarray(paired_bank[name], dtype=np.float64) for name in names}
    if r.ndim != 1 or w.shape != r.shape or np.any(r <= 0) or np.any(np.diff(r) <= 0) or np.any(w <= 0):
        raise ValueError("atomic projector construction requires increasing positive radii and dr weights")
    if not 0 < discarded_weight < 1:
        raise ValueError("discarded_weight must lie strictly between zero and one")
    if any(a.ndim != 2 or a.shape != bank["ps_u"].shape or a.shape[1] != len(r)
           or not np.all(np.isfinite(a)) for a in bank.values()):
        raise ValueError("invalid paired atomic radial bank")
    ps = bank["ps_u"]
    # This is the same Gram eigensystem, computed without squaring the
    # condition number. Core derivatives can require modes that a formed
    # normal-equations matrix already loses to cancellation.
    left, singular, right = np.linalg.svd(np.sqrt(w[:, None]) * ps.T, full_matrices=False)
    eigenvalues, eigenvectors = singular ** 2, right.T
    mass = float(np.sum(eigenvalues))
    if mass <= 0:
        raise ValueError("pseudo partial-wave bank has zero norm")
    # Sum the discarded tail directly: 1-epsilon rounds to one below the
    # floating-point unit roundoff, although its positive singular-value
    # tail is still representable and meaningful.
    tail = np.cumsum(eigenvalues[::-1])[::-1]
    rank = int(np.flatnonzero(np.r_[tail[1:], 0] <= discarded_weight * mass)[0]) + 1
    kept = eigenvalues[:rank]
    if singular[rank - 1] <= 128 * np.finfo(float).eps * singular[0]:
        raise ValueError("requested atomic reconstruction rank resolves numerical null vectors")
    coefficients = eigenvectors[:, :rank] / np.sqrt(kept)
    result = {name: values.T @ coefficients for name, values in bank.items()}
    result["ps_u"] = left[:, :rank] / np.sqrt(w[:, None])
    differences = {"delta_u": bank["ae_u"] - bank["ps_u"],
                   "delta_du_dr": bank["ae_du_dr"] - bank["ps_du_dr"]}
    boundary_error = np.hypot(differences["delta_u"][:, -1],
                              r[-1] * differences["delta_du_dr"][:, -1])
    boundary_scale = np.hypot(bank["ps_u"][:, -1], r[-1] * bank["ps_du_dr"][:, -1])
    if np.any(boundary_scale == 0) or np.max(boundary_error / boundary_scale) > 1e-10:
        raise ValueError("paired atomic waves do not match value and physical derivative at the sphere boundary")
    boundary_roundoff = float(np.max(boundary_error / boundary_scale))
    for name, values in differences.items():
        # Boundary matching defines these two differences to be exactly zero.
        # Remove only the measured root-solver roundoff before an inverse
        # singular value can amplify it into a false discontinuity.
        values[:, -1] = 0
        result[name] = values.T @ coefficients
    result["ae_u"] = result["ps_u"] + result["delta_u"]
    result["ae_du_dr"] = result["ps_du_dr"] + result["delta_du_dr"]
    result.update(coefficients=coefficients, eigenvalues=eigenvalues,
                  discarded_fraction=float(np.sum(eigenvalues[rank:]) / mass),
                  matching_boundary_roundoff=boundary_roundoff,
                  ps_gram=(result["ps_u"].T * w) @ result["ps_u"],
                  ae_gram=(result["ae_u"].T * w) @ result["ae_u"],
                  ps_ae_cross_gram=(result["ps_u"].T * w) @ result["ae_u"])
    return result


def radial_gradient_norm_squared(r: np.ndarray, weights_dr: np.ndarray,
                                 u: np.ndarray, du_dr: np.ndarray, l: int) -> np.ndarray:
    r"""Return the real-space gradient norm inside a finite atomic sphere.

    For ``u=rR`` and normalized spherical harmonics this is
    ``integral [(u'-u/r)^2 + l(l+1)(u/r)^2] dr``.  Replacing the first
    term by ``u'^2`` would omit the surface contribution for scattering
    waves, which generally do not vanish at the sphere boundary.  The
    last array dimension is radial; arbitrary leading dimensions remain.
    """
    r = np.asarray(r, dtype=np.float64)
    w = np.asarray(weights_dr, dtype=np.float64)
    values = np.asarray(u, dtype=np.float64)
    derivative = np.asarray(du_dr, dtype=np.float64)
    if (r.ndim != 1 or values.ndim < 1 or w.shape != r.shape or values.shape != derivative.shape
            or values.shape[-1] != len(r) or np.any(r <= 0) or np.any(w <= 0)
            or not all(np.all(np.isfinite(a)) for a in (r, w, values, derivative))
            or l < 0 or int(l) != l):
        raise ValueError("invalid finite-sphere gradient norm inputs")
    radial = values / r
    return np.sum(w * ((derivative - radial) ** 2 + l * (l + 1) * radial ** 2), axis=-1)


def compress_atomic_scattering_bank(bank: dict, *, discarded_weight: float) -> tuple[dict, dict]:
    """Compress training energies and audit held-out AE reconstruction.

    Each OPF column retains its ``l`` and ``kappa``.  The diagnostics compare
    held-out AE waves with ``PS + delta_OPF <PS_OPF|PS>`` in both the radial
    L2 norm and the gradient norm.  These are atomic transferability tests;
    they establish no numerical tolerance for a crystal's Sigma matrix.
    """
    r, w = bank["r"], bank["weights_dr"]
    names = ("ps_u", "ps_du_dr", "ae_u", "ae_du_dr")
    blocks, diagnostics = [], []
    for channel, (l, kappa) in enumerate(zip(bank["channel_l"], bank["channel_kappa"])):
        training = bank["training_mask"][channel]
        paired = {name: bank[name][channel, training] for name in names}
        block = optimal_radial_projectors(r, w, paired, discarded_weight=discarded_weight)
        rank = block["ps_u"].shape[1]
        ps_held = bank["ps_u"][channel, ~training]
        ae_held = bank["ae_u"][channel, ~training]
        dae_held = bank["ae_du_dr"][channel, ~training]
        coefficients = (ps_held * w) @ block["ps_u"]
        rebuilt = ps_held + coefficients @ block["delta_u"].T
        derivative = bank["ps_du_dr"][channel, ~training] + coefficients @ block["delta_du_dr"].T
        error, derivative_error = rebuilt - ae_held, derivative - dae_held
        norm = np.sum(w * ae_held ** 2, axis=1)
        error_norm = np.sum(w * error ** 2, axis=1)
        gradient_norm = radial_gradient_norm_squared(r, w, ae_held, dae_held, int(l))
        gradient_error = radial_gradient_norm_squared(r, w, error, derivative_error, int(l))
        ae_singular = np.linalg.svd(np.sqrt(w[:, None]) * block["ae_u"], compute_uv=False)
        correction_singular = np.linalg.svd(np.sqrt(w[:, None]) * block["delta_u"], compute_uv=False)
        diagnostics.append({"l": int(l), "kappa": int(kappa), "rank": rank,
                            "discarded_fraction": block["discarded_fraction"],
                            "matching_boundary_roundoff": block["matching_boundary_roundoff"],
                            "ps_gram_identity_linf": float(np.max(np.abs(block["ps_gram"] - np.eye(rank)))),
                            "ae_gram_identity_linf": float(np.max(np.abs(block["ae_gram"] - np.eye(rank)))),
                            "ae_map_operator_norm": float(ae_singular[0]),
                            "correction_map_operator_norm": float(correction_singular[0]),
                            "pseudo_bank_kept_condition": float(np.sqrt(block["eigenvalues"][0] / block["eigenvalues"][rank - 1])),
                            "heldout_relative_l2_max": float(np.max(np.sqrt(error_norm / norm))),
                            "heldout_relative_gradient_max": float(np.max(np.sqrt(gradient_error / gradient_norm))),
                            "boundary_delta_u_max": float(np.max(np.abs(block["delta_u"][-1]))),
                            "boundary_delta_du_dr_max": float(np.max(np.abs(block["delta_du_dr"][-1])))})
        if diagnostics[-1]["ps_gram_identity_linf"] > 1e-7:
            raise ValueError("atomic optimal projectors lost pseudo-metric orthonormality")
        blocks.append(block)
    arrays = {name: np.concatenate([block[name] for block in blocks], axis=1)
              for name in (*names, "delta_u", "delta_du_dr")}
    arrays.update(r=r, weights_dr=w,
                  channel_l=bank["channel_l"], channel_kappa=bank["channel_kappa"],
                  channel_nopf=np.asarray([block["ps_u"].shape[1] for block in blocks]),
                  l=np.concatenate([np.full(block["ps_u"].shape[1], l, dtype=int)
                                    for block, l in zip(blocks, bank["channel_l"])]),
                  kappa=np.concatenate([np.full(block["ps_u"].shape[1], kappa, dtype=int)
                                        for block, kappa in zip(blocks, bank["channel_kappa"])]))
    arrays["delta_R"] = arrays["delta_u"] / r[:, None]
    arrays["delta_dRdr"] = arrays["delta_du_dr"] / r[:, None] - arrays["delta_u"] / r[:, None] ** 2
    arrays["ps_R"] = arrays["ps_u"] / r[:, None]
    for index, block in enumerate(blocks):
        # Ragged channel diagnostics retain their small exact matrices without
        # padding or object arrays in the portable NPZ schema.
        for name in ("coefficients", "eigenvalues", "ps_gram", "ae_gram", "ps_ae_cross_gram"):
            arrays[f"channel_{index}_{name}"] = block[name]
    return arrays, {"discarded_weight": discarded_weight,
                    "gradient_norm_convention": "integral [(u'-u/r)^2+l(l+1)(u/r)^2] dr; u=rR",
                    "channels": diagnostics}


def scattering_branch_diagnostics(bank: dict, frozen_core: dict) -> dict:
    """Independently audit the frozen-core node offset of every matched row."""
    r = bank["r"]
    rows = []
    for channel, (l, kappa) in enumerate(zip(bank["channel_l"], bank["channel_kappa"])):
        ps = bank["ps_u"][channel]
        ps_nodes = np.sum((ps[:, 1:] * ps[:, :-1] < 0) & (r[:-1] >= 0.1), axis=1)
        core_count = np.count_nonzero(frozen_core["core_kappa"] == kappa)
        offset = bank["node_matched"][channel] - ps_nodes
        rows.append({"l": int(l), "kappa": int(kappa), "required_core_nodes": int(core_count),
                     "observed_node_offsets": np.unique(offset).tolist(),
                     "validated": bool(np.all(offset == core_count))})
    max_phase = float(np.max(np.abs(bank["boundary_phase_residual"])))
    validated = all(row["validated"] for row in rows) and max_phase < 1e-11
    return {"validated": validated, "max_unwrapped_phase_residual": max_phase,
            "channels": rows}


def write_atomic_reconstruction(path: str | Path, arrays: dict, metadata: dict) -> None:
    """Write a species sidecar and authenticate every numerical payload byte."""
    destination = Path(path)
    if destination.exists():
        raise FileExistsError(f"preserve existing atomic reconstruction: {destination}")
    metadata = dict(metadata)
    metadata["schema"] = SCHEMA
    metadata["radial_convention"] = "u=rR; derivatives=du/dr; length=bohr"
    if metadata.get("operator_comparison", {}).get("authenticated") is not True:
        raise ValueError("atomic reconstruction requires authenticated regenerated UPF operator")
    if metadata.get("phase_branch_validated") is not True:
        raise ValueError("atomic reconstruction requires a validated frozen-core phase branch")
    if "source_sha256" not in metadata:
        raise ValueError("atomic reconstruction requires source pseudopotential sha256")
    serial = {name: np.asarray(values) for name, values in arrays.items()}
    if any(not np.all(np.isfinite(values)) for values in serial.values()):
        raise ValueError("non-finite atomic reconstruction table")
    digest = hashlib.sha256()
    for name, values in sorted(serial.items()):
        digest.update(name.encode())
        digest.update(str(values.shape).encode())
        digest.update(values.dtype.str.encode())
        digest.update(values.tobytes())
    metadata["payload_sha256"] = digest.hexdigest()
    np.savez_compressed(destination, **serial,
                        metadata_json=np.asarray(json.dumps(metadata, sort_keys=True)))


def load_atomic_reconstruction(path: str | Path, source_upf: str | Path) -> dict:
    """Read a small host sidecar, refusing altered tables or a different UPF."""
    with np.load(path, allow_pickle=False) as archive:
        metadata = json.loads(str(archive["metadata_json"]))
        arrays = {name: archive[name] for name in archive.files if name != "metadata_json"}
    if metadata.get("schema") != SCHEMA:
        raise ValueError("unsupported atomic reconstruction schema")
    if metadata.get("source_sha256") != hashlib.sha256(Path(source_upf).read_bytes()).hexdigest():
        raise ValueError("atomic reconstruction belongs to a different pseudopotential")
    if metadata.get("operator_comparison", {}).get("authenticated") is not True:
        raise ValueError("atomic reconstruction lacks regenerated-operator authentication")
    if metadata.get("phase_branch_validated") is not True:
        raise ValueError("atomic reconstruction lacks frozen-core phase-branch authentication")
    digest = hashlib.sha256()
    for name, values in sorted(arrays.items()):
        digest.update(name.encode())
        digest.update(str(values.shape).encode())
        digest.update(values.dtype.str.encode())
        digest.update(values.tobytes())
    if digest.hexdigest() != metadata.get("payload_sha256"):
        raise ValueError("atomic reconstruction payload fingerprint mismatch")
    return {**arrays, "metadata": metadata}


def evaluate_radial_correction(sidecar: dict, radius: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Evaluate paired ``delta R`` and ``d(delta R)/dr`` on positive radii.

    Input ``delta_u``/``delta_du_dr`` have shape ``(n_r, n_opf)`` and the
    output has shape ``radius.shape + (n_opf,)``.  Cubic Hermite interpolation
    preserves the exported physical derivatives.  At and beyond the matching
    sphere the correction is zero.  Atomic Coulomb cusps exclude ``r=0``.
    """
    from scipy.interpolate import CubicHermiteSpline

    radii = np.asarray(radius, dtype=float)
    r = sidecar["r"]
    if np.any(radii <= 0) or np.any(radii < r[0]):
        raise ValueError("atomic correction evaluation requires r >= first positive radial sample")
    spline = CubicHermiteSpline(r, sidecar["delta_u"], sidecar["delta_du_dr"], axis=0,
                               extrapolate=False)
    inside = radii < r[-1]
    evaluation_r = np.minimum(radii, r[-1])
    u = spline(evaluation_r)
    du = spline(evaluation_r, 1)
    value = u / evaluation_r[..., None]
    derivative = du / evaluation_r[..., None] - u / evaluation_r[..., None] ** 2
    return (np.where(inside[..., None], value, 0),
            np.where(inside[..., None], derivative, 0))
